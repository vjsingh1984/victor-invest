# Architectural Issues Analysis - November 2025

## Executive Summary

Analysis of 5 reported architectural issues reveals:
- **2 issues ALREADY FIXED** (EventBus lifecycle, LLM HTTP error handling)
- **3 issues CONFIRMED VALID** (CLI config ignored, async blocking, config singleton mutation)

This document provides verification evidence and remediation guidance for the remaining issues.

---

## Issue Status Matrix

| # | Issue | Status | Severity | Lines of Code |
|---|-------|--------|----------|---------------|
| 1 | EventBus pipeline dead | ✅ **FIXED** | N/A | `orchestrator.py:248` |
| 2 | CLI config ignored | ⚠️ **VALID** | Medium | `cli_orchestrator.py:232, 267` |
| 3 | Async blocked by sync I/O | ⚠️ **VALID** | High | `cache_manager.py:218`, `agent.py:87-140` |
| 4 | LLM client ignores HTTP failures | ✅ **FIXED** | N/A | `ollama.py:200-207` |
| 5 | Shared mutable config | ⚠️ **VALID** | Medium | `cli_orchestrator.py:272-273` |

---

## Issue #1: EventBus Pipeline [ALREADY FIXED] ✅

### Original Finding
> "Event/metering pipeline is effectively dead: AgentOrchestrator.start never starts the EventBus processor"

### Verification Evidence
**File**: `src/investigator/application/orchestrator.py:248`

```python
# Start EventBus processor (Fix: EventBus pipeline was dead)
await self.event_bus.start()
self.logger.info("Event bus started - monitoring and metrics active")
```

**Status**: ✅ **Fixed in Phase 1 (Nov 2024)**

The EventBus is properly started in the orchestrator lifecycle and includes proper cleanup on shutdown (lines 256-260).

### No Action Required

---

## Issue #2: CLI Configuration Split/Ignored [VALID] ⚠️

### Finding
> "CLI configuration is split/ignored: the global click option --config is parsed but ignored by get_config()"

### Verification Evidence
**File**: `cli_orchestrator.py:232, 267`

```python
@click.pass_context
def cli(ctx, config, log_level, log_file, verbose):
    """InvestiGator - Agentic AI Investment Analysis System"""
    ctx.obj['config'] = load_config(config)  # Line 232: Loads config from --config flag

@click.pass_context
def analyze(ctx, symbol, mode, output, format, detail_level, report, force_refresh, refresh_alias):
    """Analyze a single stock symbol"""
    config = ctx.obj['config']  # Line 259: Gets loaded config from context

    async def run_analysis():
        from investigator.config import get_config
        cfg = get_config()  # Line 267: ❌ Ignores ctx config, uses singleton
```

**Status**: ⚠️ **CONFIRMED VALID**

The `--config` flag is parsed and loaded via `load_config()` but then ignored when `get_config()` is called, which returns a process-wide singleton.

### Impact
- Users cannot point CLI at alternate config files
- Testing with different configs requires environment manipulation
- Config file path flexibility is advertised but non-functional

### Remediation

**Option A: Pass config path to get_config() (Recommended)**

```python
# 1. Update get_config() signature in src/investigator/config/__init__.py
def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration singleton, optionally from specific path"""
    global _config
    if _config is None or config_path:
        _config = Config.from_yaml(config_path or 'config.yaml')
    return _config

# 2. Update CLI commands to pass config path
async def run_analysis():
    from investigator.config import get_config
    cfg = get_config(config_path=ctx.obj.get('config_path'))  # Pass path through
```

**Option B: Accept loaded config object**

```python
# Pass the loaded config dict directly to orchestrator
orchestrator = AgentOrchestrator(
    cache_manager=cache_manager,
    metrics_collector=metrics_collector,
    config=config  # Pass loaded config from ctx
)
```

**Priority**: Medium (affects testing and multi-config deployments)

---

## Issue #3: Async Blocked by Synchronous I/O [VALID] ⚠️

### Finding
> "Async stack is blocked by synchronous heavy work. Cache reads/writes hit disk/DB synchronously"

### Verification Evidence

**File**: `src/investigator/infrastructure/cache/cache_manager.py:218`

```python
def get(self, cache_type: CacheType, key: Union[Tuple, Dict]) -> Optional[Dict[str, Any]]:
    """Get data from cache (SYNCHRONOUS)"""
    for handler in handlers:
        try:
            result = handler.get(key)  # ❌ Blocking I/O call
            if result is not None:
                return result
```

**File**: `src/investigator/application/orchestrator.py:733-751` (called from async context)

```python
async def _process_task(self, task: OrchestrationTask):
    """Process a single task"""
    # ... async code ...

    # ❌ Synchronous cache call blocks event loop
    cached = self.cache_manager.get(cache_type, cache_key)

    if not cached:
        result = await agent.process(agent_task)
        # ❌ Synchronous cache write blocks event loop
        self.cache_manager.set(cache_type, cache_key, result.data)
```

**File**: `src/investigator/domain/agents/fundamental/agent.py:87-140`

```python
async def process(self, task: AgentTask) -> AgentResult:
    """Process fundamental analysis"""
    # ❌ Heavy pandas/YAML/DB work on event loop
    data = self._load_sector_multiples()  # Reads YAML from disk
    df = pd.DataFrame(quarterly_data)  # Large dataframe operations
    metrics = self._calculate_metrics(df)  # Heavy computation
```

**Status**: ⚠️ **CONFIRMED VALID - HIGH SEVERITY**

### Impact
- **High**: One slow cache write (e.g., 200ms disk I/O) blocks ALL agent coroutines
- **Concurrency ceiling**: Even with 10 agent slots, one blocking call = 9 idle agents
- **Cascading delays**: VRAM semaphore can't help if event loop is blocked
- **Performance cliff**: System performs well until first cache miss, then everything stalls

### Remediation

**Option A: Make cache handlers async (Preferred for correctness)**

```python
# 1. Update cache handler interface
class CacheHandler(ABC):
    @abstractmethod
    async def get(self, key: Any) -> Optional[Dict]:
        pass

    @abstractmethod
    async def set(self, key: Any, value: Dict, ttl: Optional[int] = None):
        pass

# 2. Update file handler with async I/O
class FileCacheHandler(CacheHandler):
    async def get(self, key: Any) -> Optional[Dict]:
        path = self._key_to_path(key)
        if path.exists():
            async with aiofiles.open(path, 'r') as f:  # Use aiofiles
                content = await f.read()
                return json.loads(content)
        return None

# 3. Update CacheManager to be async
class CacheManager:
    async def get(self, cache_type: CacheType, key: Union[Tuple, Dict]) -> Optional[Dict]:
        for handler in handlers:
            result = await handler.get(key)  # Now async
            if result is not None:
                return result
        return None
```

**Option B: Offload to thread pool (Simpler short-term fix)**

```python
# Update CacheManager to use thread executor
class CacheManager:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache")

    async def get(self, cache_type: CacheType, key: Union[Tuple, Dict]) -> Optional[Dict]:
        """Async wrapper that offloads sync I/O to thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_get,  # Existing sync implementation
            cache_type,
            key
        )
```

**Option C: Hybrid approach (Best of both)**

```python
# Fast path: Memory cache (async)
# Slow path: Disk/DB cache (thread pool)
class CacheManager:
    async def get(self, cache_type: CacheType, key: Union[Tuple, Dict]) -> Optional[Dict]:
        # Try memory cache first (instant)
        if result := self._memory_cache.get(key):
            return result

        # Offload disk/DB to thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_get_from_storage,
            cache_type,
            key
        )
```

**Priority**: High (affects system-wide concurrency and performance)

---

## Issue #4: LLM Client Ignores HTTP Failures [ALREADY FIXED] ✅

### Original Finding
> "LLM client ignores HTTP failures and can return garbage on non-200s"

### Verification Evidence
**File**: `src/investigator/infrastructure/llm/ollama.py:200-207`

```python
async def pull_model(self, model: str, stream: bool = True) -> AsyncIterator[Dict]:
    """Pull a model from Ollama registry"""
    async with self._session.post(endpoint, json=payload) as response:
        # Fix: Check HTTP status before parsing JSON
        if response.status != 200:
            error_text = await response.text()
            raise OllamaHTTPError(
                f"Failed to pull model '{model}': {error_text}",
                status_code=response.status,
                endpoint=endpoint,
                model=model
            )
```

**Status**: ✅ **Fixed in Phase 1 (Nov 2024)**

The LLM client now properly checks `response.status` before parsing JSON and raises descriptive exceptions with model/endpoint context.

### No Action Required

---

## Issue #5: Shared Mutable Config [VALID] ⚠️

### Finding
> "Shared mutable config toggles risk cross-run leakage: analyze/batch mutate the singleton cfg.cache_control.force_refresh"

### Verification Evidence
**File**: `cli_orchestrator.py:272-273`

```python
async def run_analysis():
    from investigator.config import get_config
    cfg = get_config()  # Global singleton

    original_force_refresh = getattr(cfg.cache_control, 'force_refresh', False)
    original_force_symbols = getattr(cfg.cache_control, 'force_refresh_symbols', None)

    if force_refresh:
        cfg.cache_control.force_refresh = True  # ❌ Mutates global state
        cfg.cache_control.force_refresh_symbols = [symbol]  # ❌ Affects other runs
```

**Status**: ⚠️ **CONFIRMED VALID**

### Impact
- **Concurrency hazard**: Concurrent CLI runs (or API + CLI) see each other's cache flags
- **State leakage**: A `--force-refresh` run can affect subsequent runs if cleanup fails
- **Test pollution**: Unit tests that mutate config affect subsequent tests

### Remediation

**Option A: Immutable config with per-run overrides (Recommended)**

```python
@dataclass(frozen=True)  # Make config immutable
class CacheControl:
    use_cache: bool = True
    read_from_cache: bool = True
    # Remove mutable fields, use method overrides instead

    def with_force_refresh(self, symbols: List[str]) -> 'CacheControl':
        """Return new config with force refresh for symbols"""
        return dataclasses.replace(
            self,
            use_cache=False,  # Or keep as override dict
        )

# In CLI:
async def run_analysis():
    cfg = get_config()
    if force_refresh:
        cache_overrides = {'force_refresh': True, 'symbols': [symbol]}
        orchestrator = AgentOrchestrator(
            config=cfg,
            cache_overrides=cache_overrides  # Pass as parameter
        )
```

**Option B: Context-local config copies**

```python
async def run_analysis():
    cfg = get_config()

    # Create isolated copy for this run
    run_config = copy.deepcopy(cfg)
    if force_refresh:
        run_config.cache_control.force_refresh = True
        run_config.cache_control.force_refresh_symbols = [symbol]

    orchestrator = AgentOrchestrator(
        config=run_config  # Each run gets isolated config
    )
```

**Option C: Cache manager overrides (Minimal change)**

```python
# Don't mutate config, pass overrides to cache manager
cache_manager = get_cache_manager()
if force_refresh:
    cache_manager.set_override(
        force_refresh=True,
        symbols=[symbol]
    )

try:
    # Run analysis
    result = await orchestrator.analyze(symbol)
finally:
    # Clear overrides after run
    cache_manager.clear_overrides()
```

**Priority**: Medium (affects concurrent/API usage and test isolation)

---

## Remediation Priority and Effort

| Issue | Priority | Effort | Risk |
|-------|----------|--------|------|
| #3 Async Blocking | **High** | Medium (Option B: 2-3 days) | Low |
| #2 CLI Config | Medium | Small (1 day) | Low |
| #5 Config Mutation | Medium | Small-Medium (1-2 days) | Low |

### Recommended Implementation Order

1. **Issue #3 (Async Blocking)** - Option B (thread pool) first
   - Immediate performance improvement
   - Low risk, well-tested pattern
   - Can later migrate to Option A (full async) when time permits

2. **Issue #2 (CLI Config)** - Option A (pass path to get_config)
   - Enables testing with multiple configs
   - Required for proper integration tests

3. **Issue #5 (Config Mutation)** - Option C (cache manager overrides)
   - Minimal code change
   - Maintains backward compatibility
   - Can refactor to Option A later if needed

---

## Testing Strategy for Fixes

### Issue #3 (Async Blocking) Testing

```python
# Test that cache I/O doesn't block event loop
async def test_cache_concurrent_access():
    """Verify multiple agents can cache concurrently"""
    cache_manager = CacheManager()

    # Simulate 10 concurrent cache writes
    tasks = [
        cache_manager.set(CacheType.LLM_RESPONSE, f"key_{i}", {"data": f"value_{i}"})
        for i in range(10)
    ]

    start = time.time()
    await asyncio.gather(*tasks)
    elapsed = time.time() - start

    # Should complete in ~200ms, not 2000ms (10 × 200ms sequential)
    assert elapsed < 0.5, f"Cache writes blocked: {elapsed}s"
```

### Issue #2 (CLI Config) Testing

```python
def test_cli_respects_config_flag(tmp_path):
    """Verify --config flag is honored"""
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text("database:\n  host: custom-host")

    result = runner.invoke(cli, ['--config', str(custom_config), 'analyze', 'AAPL'])

    # Verify custom config was used (check logs or behavior)
    assert "custom-host" in result.output
```

### Issue #5 (Config Mutation) Testing

```python
async def test_concurrent_runs_isolated():
    """Verify concurrent runs don't interfere"""
    async def run_with_force_refresh():
        result = await orchestrator.analyze('AAPL', force_refresh=True)
        return result

    async def run_normal():
        result = await orchestrator.analyze('MSFT', force_refresh=False)
        return result

    # Run concurrently
    results = await asyncio.gather(
        run_with_force_refresh(),
        run_normal()
    )

    # Verify MSFT used cache despite AAPL force refresh
    assert results[1].metadata['cache_hit'] == True
```

---

## Implementation Checklist

### Phase 1: Critical Fix (Week 1)
- [ ] Implement Issue #3 (Async Blocking) - Option B (thread pool)
  - [ ] Add ThreadPoolExecutor to CacheManager
  - [ ] Wrap sync I/O in run_in_executor
  - [ ] Test concurrent cache access
  - [ ] Measure performance improvement
  - [ ] Update CLAUDE.md to remove "Issue #3" from known issues

### Phase 2: Configuration Fixes (Week 2)
- [ ] Implement Issue #2 (CLI Config) - Option A
  - [ ] Update get_config() to accept path parameter
  - [ ] Thread path through CLI commands
  - [ ] Test with alternate config files
  - [ ] Update CLAUDE.md configuration section

- [ ] Implement Issue #5 (Config Mutation) - Option C
  - [ ] Add override mechanism to CacheManager
  - [ ] Update CLI to use overrides instead of mutation
  - [ ] Test concurrent runs
  - [ ] Update CLAUDE.md testing guidelines

### Phase 3: Documentation (Week 2)
- [ ] Update docs/ARCHITECTURE.md with async patterns
- [ ] Update docs/OPERATIONS_RUNBOOK.md with new testing commands
- [ ] Add performance benchmarks to docs/
- [ ] Update CLAUDE.md "Known Architectural Issues" section

---

## Conclusion

Out of 5 reported issues:
- **2 are already fixed** (EventBus, LLM errors) - evidence of active maintenance
- **3 remain valid** but all are addressable with low-to-medium effort

The async blocking issue (#3) is the most impactful and should be addressed first using the thread pool approach (Option B) for quick wins, with a migration path to full async I/O later.

The codebase shows good architectural discipline with the fixes already in place, and the remaining issues are typical for systems evolving from synchronous to async patterns.
