#!/usr/bin/env python3
"""
Resource-Aware Ollama Client Pool v3
- Uses Ollama REST API (/api/ps) for actual VRAM usage
- Per-server locking to prevent race conditions
- Accurate memory estimates (accounts for KV cache + buffers)

ASCII map of the moving parts:

+------------------+      reserve/release      +-----------------------+
|  Agent Request   | ------------------------> | ResourceAwareOllamaPool|
|  (prompt text)   |                           |  (global view)         |
+------------------+ <------------------------ |                       |
        |                 status updates       +----------+------------+
        |                                                   |
        | acquire model                                     | per-server lock
        v                                                   v
+------------------+     poll /api/ps     +----------------------+
| ServerStatus     | <------------------->| Ollama Server (HTTP) |
| (per endpoint)   |                      +----------------------+
+------------------+

The pool keeps pessimistic reservations so concurrent tasks never oversubscribe VRAM.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

from investigator.infrastructure.llm.vram_calculator import (
    estimate_kv_cache_only,
    estimate_model_vram_requirement,
)

logger = logging.getLogger(__name__)


class PoolStrategy(Enum):
    """Load balancing strategies"""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_BUSY = "least_busy"
    MOST_CAPACITY = "most_capacity"
    PREFER_REMOTE = "prefer_remote"  # Prefer non-localhost servers first (better hardware)


@dataclass
class ServerCapacity:
    """Server hardware capacity"""

    url: str
    total_ram_gb: int
    usable_ram_gb: int
    metal: bool = True
    max_concurrent: int = 1  # Changed default to 1 (one 32B model per 48GB)
    priority: int = 0  # Higher = preferred (0=default, 100=highest priority for remote M4 Max)


@dataclass
class RunningModel:
    """Model currently running on a server"""

    name: str
    size: int
    size_vram: int  # ACTUAL VRAM usage (includes KV cache!)
    digest: str
    expires_at: Optional[str] = None


@dataclass
class ServerStatus:
    """Track real-time status of an Ollama server"""

    url: str
    capacity: ServerCapacity

    # Runtime state
    active_requests: int = 0
    total_requests: int = 0
    failures: int = 0
    last_used: Optional[datetime] = None
    available: bool = True

    # Real-time state from /api/ps
    running_models: List[RunningModel] = field(default_factory=list)
    total_vram_used_gb: float = 0.0

    # Pessimistic reservation (for requests in flight)
    reserved_ram_gb: float = 0.0

    @property
    def free_ram_gb(self) -> float:
        """Calculate free RAM (actual usage + reservations)"""
        return max(0.0, self.capacity.usable_ram_gb - self.total_vram_used_gb - self.reserved_ram_gb)

    @property
    def ram_utilization(self) -> float:
        """RAM utilization percentage"""
        if self.capacity.usable_ram_gb == 0:
            return 0.0
        used = self.total_vram_used_gb + self.reserved_ram_gb
        return (used / self.capacity.usable_ram_gb) * 100

    def can_load_model(self, model_memory_gb: float) -> bool:
        """Check if server has capacity to load a model"""
        return self.free_ram_gb >= model_memory_gb


class ResourceAwareOllamaPool:
    """
    Resource-aware connection pool for Ollama servers v3

    Improvements over v2:
    - Per-server locking (no race conditions)
    - Pessimistic memory reservation
    - Accurate memory estimates (model + KV cache + buffers)
    """

    def __init__(
        self,
        servers: List[ServerCapacity],
        model_specs: Dict[str, Any] = None,
        strategy: PoolStrategy = PoolStrategy.MOST_CAPACITY,
        max_failures: int = 3,
        timeout: int = 300,
        max_prompt_tokens: int = 32768,
    ):
        self.servers = {s.url: ServerStatus(url=s.url, capacity=s) for s in servers}
        self.model_specs = model_specs or {}
        self.strategy = strategy
        self.max_failures = max_failures
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_prompt_tokens = max(1024, max_prompt_tokens)

        # Round-robin state
        self.current_index = 0
        self.server_list = list(self.servers.keys())

        # Concurrency control
        self.lock = asyncio.Lock()
        self.server_locks = {url: asyncio.Lock() for url in self.servers.keys()}
        self.capacity_available = asyncio.Condition(self.lock)  # Wait/notify for capacity changes

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(
            "🏁 POOL_INIT servers=%d strategy=%s max_prompt_tokens=%d",
            len(servers),
            self.strategy.value,
            self.max_prompt_tokens,
        )
        for server in servers:
            priority_str = f" priority={server.priority}" if server.priority > 0 else ""
            logger.debug(
                f"🏁 POOL_SERVER url={server.url} usable={server.usable_ram_gb:.1f}GB max_concurrent={server.max_concurrent} metal={server.metal}{priority_str}"
            )
            # Old format for compatibility
            logger.debug(
                "🏁 POOL_SERVER url=%s usable=%.1fGB max_concurrent=%d metal=%s",
                server.url,
                server.usable_ram_gb,
                server.max_concurrent,
                server.metal,
            )

    def _get_model_spec(self, model_name: str) -> Optional[Any]:
        return self.model_specs.get(model_name)

    def _spec_value(self, spec: Any, attr: str, default: Any) -> Any:
        if spec is None:
            return default
        if hasattr(spec, attr):
            return getattr(spec, attr)
        if isinstance(spec, dict):
            return spec.get(attr, default)
        return default

    def _estimate_tokens(self, text: Optional[str]) -> int:
        if not text:
            return 0
        length = len(text)
        return max(1, length // 4 + 1)

    def _estimate_kv_cache_gb(self, spec: Any, prompt_tokens: int, response_tokens: int) -> float:
        per_1k = float(self._spec_value(spec, "kv_cache_mb_per_1k_tokens", 120.0))
        total_tokens = max(0, prompt_tokens + response_tokens)
        return (total_tokens / 1000.0) * per_1k / 1024.0

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self):
        if not self._session:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def initialize_servers(self) -> None:
        """
        Perform an initial health-check against all configured servers.

        Any endpoint that fails the /api/ps probe is marked unavailable and
        removed from the active dispatch list before work begins.
        """
        if not self.server_list:
            return

        # Warm up the session so we reuse connections during initialization.
        await self._ensure_session()

        # Query each server in parallel; failures are captured inside update_server_status.
        await asyncio.gather(
            *(self.update_server_status(url) for url in list(self.server_list)),
            return_exceptions=True,
        )

        removed: List[str] = []
        async with self.lock:
            for url in list(self.server_list):
                server = self.servers.get(url)
                if not server or not server.available:
                    removed.append(url)
                    self.servers.pop(url, None)
                    self.server_locks.pop(url, None)
            if removed:
                self.server_list = [url for url in self.server_list if url not in removed]

        if removed:
            logger.warning(
                "POOL_INIT_REMOVE removed unreachable servers: %s",
                ", ".join(removed),
            )

    async def get_server_status(self, server_url: str) -> Dict[str, Any]:
        """Query server for current resource usage via /api/ps"""
        await self._ensure_session()
        try:
            async with self._session.get(f"{server_url}/api/ps") as response:
                if response.status == 200:
                    return await response.json()
                logger.warning(
                    "POOL_HEALTH http_error url=%s status=%s",
                    server_url,
                    response.status,
                )
                return {"models": [], "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.warning("POOL_HEALTH connection_error url=%s error=%s", server_url, e)
            return {"models": [], "error": str(e)}

    async def update_server_status(self, server_url: str):
        """Update server status by querying /api/ps"""
        status_data = await self.get_server_status(server_url)

        async with self.lock:
            if server_url in self.servers:
                server = self.servers[server_url]

                if status_data.get("error"):
                    server.available = False
                    server.running_models = []
                    server.total_vram_used_gb = 0.0
                    logger.warning(
                        "POOL_SERVER_UNAVAILABLE url=%s error=%s",
                        server_url,
                        status_data.get("error"),
                    )
                    return

                # Parse running models
                running_models = []
                total_vram_bytes = 0

                for model_data in status_data.get("models", []):
                    running_model = RunningModel(
                        name=model_data["name"],
                        size=model_data["size"],
                        size_vram=model_data.get("size_vram", 0),  # ACTUAL VRAM!
                        digest=model_data["digest"],
                        expires_at=model_data.get("expires_at"),
                    )
                    running_models.append(running_model)
                    total_vram_bytes += running_model.size_vram

                # Update server status with ACTUAL usage
                if server.failures >= self.max_failures:
                    server.available = False
                    logger.debug(
                        "POOL_SERVER_STILL_UNAVAILABLE url=%s failures=%d >= max_failures=%d",
                        server_url,
                        server.failures,
                        self.max_failures,
                    )
                else:
                    server.available = True
                server.running_models = running_models
                server.total_vram_used_gb = total_vram_bytes / (1024**3)

                logger.debug(
                    f"Server {server_url}: {len(running_models)} models loaded, "
                    f"{server.total_vram_used_gb:.1f}GB VRAM used, "
                    f"{server.reserved_ram_gb:.1f}GB reserved, "
                    f"{server.free_ram_gb:.1f}GB free"
                )

    def get_model_memory_estimate(self, model_name: str) -> float:
        """
        Estimate BASE model memory (weights only, no KV cache)

        Returns the base model weight size. KV cache and overhead are
        calculated separately in select_server_for_model() based on
        actual context length.

        Note: This returns JUST the model weights, not inference memory.
        """
        spec = self._get_model_spec(model_name)
        if spec is not None:
            return float(self._spec_value(spec, "weights_vram_gb", self._spec_value(spec, "memory_gb", 0.0)))

        # PRIORITY 2: Name-based estimates for BASE model weights only
        model_lower = model_name.lower()
        if "70b" in model_lower:
            return 42.0  # Base weights for 70B models
        elif "32b" in model_lower or "33b" in model_lower:
            return 19.0  # Base weights for 32B models
        elif "30b" in model_lower:
            return 16.2  # Base weights for 30B models
        elif "14b" in model_lower:
            return 8.0  # Base weights for 14B models
        elif "8b" in model_lower:
            return 4.5  # Base weights for 8B models
        elif "7b" in model_lower:
            return 4.0  # Base weights for 7B models

        # Conservative default - base weights only
        logger.warning(f"No memory estimate for {model_name}, using default 20GB base weight")
        return 20.0

    async def select_server_for_model(
        self, model_name: str, context_tokens: int, spec: Any = None
    ) -> Optional[tuple[str, float, bool]]:
        """
        Select server with sufficient capacity for model

        Updates all servers via /api/ps first
        """
        # Update all server statuses in parallel
        await asyncio.gather(*[self.update_server_status(url) for url in self.servers.keys()])

        spec = spec or self._get_model_spec(model_name)

        # Use centralized VRAM calculator (single source of truth)
        vram_est = estimate_model_vram_requirement(spec, include_kv_cache=True)
        required_new = vram_est["total_gb"]
        required_reuse = vram_est["kv_cache_gb"]  # When reusing loaded model, only need KV cache

        # Wait/notify pattern: wait for capacity with timeout
        max_wait_seconds = 600  # 10 minutes max wait
        start_time = time.time()
        wait_logged = False

        async with self.capacity_available:
            while True:
                # Check for available capacity
                reuse_candidates = []
                new_candidates = []
                for server in self.servers.values():
                    has_model = any(m.name == model_name for m in server.running_models)
                    required = required_reuse if has_model else required_new
                    if (
                        server.available
                        and server.active_requests < server.capacity.max_concurrent
                        and server.can_load_model(required)
                    ):
                        if has_model:
                            reuse_candidates.append((server, required))
                        else:
                            new_candidates.append((server, required))

                candidate_pool: List[tuple[ServerStatus, float, bool]]  # (server, vram, is_reuse)

                # For ROUND_ROBIN, RANDOM, MOST_CAPACITY, and PREFER_REMOTE: combine pools for distribution
                # MOST_CAPACITY will naturally prefer servers with more free RAM
                # PREFER_REMOTE will prefer by priority then RAM, regardless of reuse
                if self.strategy in (
                    PoolStrategy.ROUND_ROBIN,
                    PoolStrategy.RANDOM,
                    PoolStrategy.MOST_CAPACITY,
                    PoolStrategy.PREFER_REMOTE,
                ):
                    candidate_pool = [(s, r, True) for s, r in reuse_candidates] + [
                        (s, r, False) for s, r in new_candidates
                    ]
                # For LEAST_BUSY: prefer reuse but allow new if needed
                elif reuse_candidates:
                    candidate_pool = [(s, r, True) for s, r in reuse_candidates]
                else:
                    candidate_pool = [(s, r, False) for s, r in new_candidates]

                # Found capacity - proceed
                if candidate_pool:
                    if self.strategy == PoolStrategy.MOST_CAPACITY:
                        selected, required, reuse_existing = max(candidate_pool, key=lambda item: item[0].free_ram_gb)
                    elif self.strategy == PoolStrategy.PREFER_REMOTE:
                        # Sort by priority (descending), then by free RAM (descending)
                        # Higher priority servers (remote M4 Max) preferred, then most available RAM
                        sorted_pool = sorted(
                            candidate_pool,
                            key=lambda item: (item[0].capacity.priority, item[0].free_ram_gb),
                            reverse=True,
                        )
                        selected, required, reuse_existing = sorted_pool[0]
                    elif self.strategy == PoolStrategy.LEAST_BUSY:
                        selected, required, reuse_existing = min(
                            candidate_pool, key=lambda item: item[0].active_requests
                        )
                    elif self.strategy == PoolStrategy.ROUND_ROBIN:
                        index = self.current_index % len(candidate_pool)
                        selected, required, reuse_existing = candidate_pool[index]
                        self.current_index += 1
                    elif self.strategy == PoolStrategy.RANDOM:
                        selected, required, reuse_existing = random.choice(candidate_pool)
                    else:
                        selected, required, reuse_existing = candidate_pool[0]

                    if wait_logged:
                        logger.info(
                            "✅ POOL_CAPACITY_AVAILABLE model=%s waited=%.1fs", model_name, time.time() - start_time
                        )

                    logger.info(
                        "🚀 POOL_DISPATCH server=%s model=%s reuse=%s request_vram=%.2fGB "
                        "context_tokens=%d free_before=%.1fGB active=%d/%d running=%s",
                        selected.url,
                        model_name,
                        reuse_existing,
                        required,
                        context_tokens,
                        selected.free_ram_gb,
                        selected.active_requests,
                        selected.capacity.max_concurrent,
                        [m.name for m in selected.running_models],
                    )
                    return selected.url, required, reuse_existing

                # No capacity - check timeout
                elapsed = time.time() - start_time
                if elapsed >= max_wait_seconds:
                    logger.error(
                        "POOL_DISPATCH_TIMEOUT model=%s required_vram=%.2fGB waited=%.1fs summary=%s",
                        model_name,
                        required_new,
                        elapsed,
                        await self._get_server_summary(),
                    )
                    return None

                # Log once when we start waiting
                if not wait_logged:
                    logger.warning(
                        "⏳ POOL_WAITING model=%s required_vram=%.2fGB summary=%s",
                        model_name,
                        required_new,
                        await self._get_server_summary(),
                    )
                    wait_logged = True

                # Wait for notification (released when capacity freed)
                try:
                    await asyncio.wait_for(self.capacity_available.wait(), timeout=max(1.0, max_wait_seconds - elapsed))
                except asyncio.TimeoutError:
                    # Continue loop to check timeout
                    pass

    async def mark_request_start(self, server_url: str, model_memory_gb: float):
        """
        Mark that a request started and reserve memory

        Pessimistically reserves memory to prevent race conditions
        """
        async with self.lock:
            if server_url in self.servers:
                self.servers[server_url].active_requests += 1
                self.servers[server_url].total_requests += 1
                self.servers[server_url].reserved_ram_gb += model_memory_gb
                self.servers[server_url].reserved_ram_gb = max(0.0, self.servers[server_url].reserved_ram_gb)
                self.servers[server_url].last_used = datetime.now()

                logger.debug(
                    f"Reserved {model_memory_gb:.1f}GB on {server_url}, "
                    f"total reserved: {self.servers[server_url].reserved_ram_gb:.1f}GB"
                )

    async def mark_request_end(self, server_url: str, model_memory_gb: float, success: bool = True):
        """
        Mark that a request finished and release reservation

        Updates actual VRAM usage via /api/ps
        """
        async with self.lock:
            if server_url in self.servers:
                server_status = self.servers[server_url]
                server_status.active_requests -= 1
                server_status.reserved_ram_gb -= model_memory_gb
                server_status.active_requests = max(0, server_status.active_requests)
                server_status.reserved_ram_gb = max(0.0, server_status.reserved_ram_gb)

                if success:
                    if server_status.failures > 0:
                        logger.debug(
                            "POOL_RECOVERY server=%s resetting failure count after successful request",
                            server_url,
                        )
                    server_status.failures = 0
                    server_status.available = True
                else:
                    server_status.failures += 1

                    if server_status.failures >= self.max_failures:
                        server_status.available = False
                        logger.warning(
                            "Server %s marked unavailable after %d consecutive failures",
                            server_url,
                            server_status.failures,
                        )

        # Update actual VRAM usage after request
        await self.update_server_status(server_url)

        # Notify waiting tasks that capacity may be available
        async with self.capacity_available:
            self.capacity_available.notify_all()

    async def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate completion using server with sufficient capacity

        Uses per-server locking to prevent race conditions
        """
        spec = self._get_model_spec(model)
        system_prompt = kwargs.get("system") or ""
        config_obj = kwargs.get("config")

        num_predict = None
        if config_obj is not None and hasattr(config_obj, "num_predict"):
            num_predict = config_obj.num_predict
        if not num_predict:
            num_predict = int(self._spec_value(spec, "default_num_predict", 1024))

        context_limit_spec = int(self._spec_value(spec, "context_window", self.max_prompt_tokens))
        context_limit = max(1024, min(self.max_prompt_tokens, context_limit_spec))

        prompt_tokens = self._estimate_tokens(prompt) + self._estimate_tokens(system_prompt)
        prompt_tokens = min(prompt_tokens, context_limit)
        response_tokens = max(1, min(num_predict, max(0, context_limit - prompt_tokens)))

        total_tokens_for_cache = max(prompt_tokens + response_tokens, 1024)
        total_tokens_for_cache = min(total_tokens_for_cache, context_limit)

        selection = await self.select_server_for_model(model, total_tokens_for_cache, spec)
        if not selection:
            raise RuntimeError(
                f"No server has capacity for {model} "
                f"(req≈{self._estimate_kv_cache_gb(spec, prompt_tokens, response_tokens):.1f}GB KV cache)"
            )

        server_url, request_vram, reuse_existing = selection

        request_vram = max(0.0, request_vram)

        # Acquire per-server lock (prevents concurrent loading on same server)
        async with self.server_locks[server_url]:
            await self.mark_request_start(server_url, request_vram)

            try:
                from investigator.infrastructure.llm.ollama import OllamaClient

                async with OllamaClient(base_url=server_url) as client:
                    response = await client.generate(model, prompt, **kwargs)

                await self.mark_request_end(server_url, request_vram, success=True)
                return response

            except Exception as e:
                await self.mark_request_end(server_url, request_vram, success=False)
                # Temporarily mark server unavailable to avoid tight retry loop
                async with self.lock:
                    if server_url in self.servers:
                        self.servers[server_url].available = False
                        logger.warning(
                            "POOL_SERVER_MARKED_UNAVAILABLE url=%s reason=request_failure",
                            server_url,
                        )
                logger.error(f"Request failed on {server_url}: {e}")

                # Retry on another server
                return await self.generate(model, prompt, **kwargs)

    async def _get_server_summary(self) -> str:
        """Get summary of all servers for logging"""
        summaries = []
        for s in self.servers.values():
            summaries.append(
                f"{s.url}: {s.total_vram_used_gb:.1f}GB used + {s.reserved_ram_gb:.1f}GB reserved / "
                f"{s.capacity.usable_ram_gb}GB total ({s.ram_utilization:.0f}%), "
                f"{len(s.running_models)} models, {s.active_requests} active"
            )
        return "; ".join(summaries)

    async def get_pool_status(self) -> Dict[str, Any]:
        """Get detailed status of all servers"""
        await asyncio.gather(*[self.update_server_status(url) for url in self.servers.keys()])

        async with self.lock:
            return {
                "servers": [
                    {
                        "url": s.url,
                        "available": s.available,
                        "capacity": {
                            "total_ram_gb": s.capacity.total_ram_gb,
                            "usable_ram_gb": s.capacity.usable_ram_gb,
                            "metal": s.capacity.metal,
                            "max_concurrent": s.capacity.max_concurrent,
                        },
                        "usage": {
                            "vram_used_gb": round(s.total_vram_used_gb, 1),
                            "reserved_gb": round(s.reserved_ram_gb, 1),
                            "free_ram_gb": round(s.free_ram_gb, 1),
                            "utilization_pct": round(s.ram_utilization, 1),
                            "active_requests": s.active_requests,
                            "total_requests": s.total_requests,
                        },
                        "running_models": [
                            {"name": m.name, "vram_gb": round(m.size_vram / (1024**3), 1), "expires_at": m.expires_at}
                            for m in s.running_models
                        ],
                        "failures": s.failures,
                    }
                    for s in self.servers.values()
                ],
                "strategy": self.strategy.value,
                "total_servers": len(self.servers),
                "available_servers": sum(1 for s in self.servers.values() if s.available),
                "total_capacity_gb": sum(s.capacity.usable_ram_gb for s in self.servers.values()),
                "used_capacity_gb": round(
                    sum(s.total_vram_used_gb + s.reserved_ram_gb for s in self.servers.values()), 1
                ),
            }


def create_resource_aware_pool(config) -> ResourceAwareOllamaPool:
    """Create ResourceAwareOllamaPool from configuration"""
    ollama_config = config.ollama

    analysis_max_tokens = getattr(config.analysis, "max_prompt_tokens", 32768)

    if hasattr(ollama_config, "servers") and ollama_config.servers:
        servers = []
        for server_config in ollama_config.servers:
            if isinstance(server_config, dict):
                servers.append(
                    ServerCapacity(
                        url=server_config["url"],
                        total_ram_gb=server_config.get("total_ram_gb", 64),
                        usable_ram_gb=server_config.get("usable_ram_gb", 48),
                        metal=server_config.get("metal", True),
                        max_concurrent=server_config.get("max_concurrent", 1),  # Default 1
                        priority=server_config.get("priority", 0),  # Default 0 (localhost), 100 for remote M4 Max
                    )
                )
            else:
                servers.append(
                    ServerCapacity(url=server_config, total_ram_gb=64, usable_ram_gb=48, max_concurrent=1, priority=0)
                )

        strategy_str = getattr(ollama_config, "pool_strategy", "most_capacity")
        strategy = PoolStrategy(strategy_str)

        model_specs = {}
        if hasattr(ollama_config, "model_specs"):
            model_specs = ollama_config.model_specs

        logger.info(f"Creating resource-aware Ollama pool v3 with {len(servers)} servers")
        return ResourceAwareOllamaPool(servers, model_specs, strategy=strategy, max_prompt_tokens=analysis_max_tokens)

    else:
        base_url = getattr(ollama_config, "base_url", "http://localhost:11434")
        server = ServerCapacity(url=base_url, total_ram_gb=64, usable_ram_gb=48, max_concurrent=1)

        model_specs = {}
        if hasattr(ollama_config, "model_specs"):
            model_specs = ollama_config.model_specs

        logger.info(f"Creating resource-aware pool v3 with single server: {base_url}")
        return ResourceAwareOllamaPool([server], model_specs, max_prompt_tokens=analysis_max_tokens)
