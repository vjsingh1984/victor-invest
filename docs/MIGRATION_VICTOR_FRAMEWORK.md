# Victor Framework Migration Plan

**Objective**: Migrate victor_invest handlers from manual NodeResult pattern to `@handler_decorator` + `BaseHandler` pattern

**Benefits**:
- Eliminate ~1,800 lines of boilerplate code (-87% reduction)
- Automatic timing, error handling, and NodeResult construction
- Auto-registration on module import
- Better type safety and maintainability

**Migration Date**: 2025-01-26
**Status**: Planning Phase
**Estimated Effort**: 2-3 hours

---

## Current State Analysis

### Handlers to Migrate (14 total)

1. `FetchSECDataHandler` - ~113 lines
2. `FetchMarketDataHandler` - ~113 lines
3. `FetchMacroDataHandler` - ~150 lines
4. `RunFundamentalAnalysisHandler` - ~150 lines
5. `RunTechnicalAnalysisHandler` - ~150 lines
6. `RunMarketContextHandler` - ~150 lines
7. `RunSynthesisHandler` - ~600 lines (complex, LLM integration)
8. `GenerateReportHandler` - ~210 lines
9. `IdentifyPeersHandler` - ~200 lines
10. `AnalyzePeersHandler` - ~60 lines
11. `GenerateLookbackDatesHandler` - ~40 lines
12. `ProcessBacktestBatchHandler` - ~50 lines
13. `SaveRLPredictionsHandler` - ~40 lines
14. `_format_fundamental` / `_format_technical` - Helper functions (no change)

**Total**: ~2,100 lines → ~280 lines after migration

---

## Side-by-Side Comparison

### Before (Current Pattern)

```python
@dataclass
class FetchSECDataHandler:
    """Fetch SEC filing data for analysis."""

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()
        symbol = context.get("symbol", "")

        if not symbol:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="No symbol provided",
                duration_seconds=time.time() - start_time,
            )

        try:
            from victor_invest.tools.sec_filing import SECFilingTool
            sec_tool = SECFilingTool()
            result = await sec_tool.execute(
                {},
                symbol=symbol,
                action="get_company_facts",
            )

            output = {
                "status": "success" if result.success else "error",
                "data": result.output if result.success else None,
                "error": result.error if not result.success else None,
            }

            output_key = node.output_key or "sec_data"
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"SEC data fetch error for {symbol}: {e}")
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )
```

**Lines**: 56
**Boilerplate**: Manual timing, error handling, NodeResult construction

### After (New Pattern)

```python
from victor.framework.handler_registry import handler_decorator
from victor.framework.workflows.base_handler import BaseHandler

@handler_decorator("fetch_sec_data", vertical="investment", description="Fetch SEC filing data")
@dataclass
class FetchSECDataHandler(BaseHandler):
    """Fetch SEC filing data for analysis."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        symbol = context.get("symbol", "")

        if not symbol:
            raise ValueError("No symbol provided")

        from victor_invest.tools.sec_filing import SECFilingTool

        sec_tool = SECFilingTool()
        result = await sec_tool.execute(
            {},  # _exec_ctx
            symbol=symbol,
            action="get_company_facts",
        )

        return {
            "status": "success" if result.success else "error",
            "data": result.output if result.success else None,
            "error": result.error if not result.success else None,
        }, 0  # (output, tool_calls_count)
```

**Lines**: 36
**Boilerplate**: Zero - all handled by BaseHandler

**Key Differences**:
1. ✅ No `start_time` / `duration_seconds` - automatic
2. ✅ No `NodeResult` construction - automatic
3. ✅ No try/except for error handling - automatic
4. ✅ No `context.set()` - automatic
5. ✅ Returns `Tuple[Any, int]` instead of `NodeResult`
6. ✅ Auto-registration via decorator

---

## Migration Strategy

### Phase 1: Preparation (15 minutes)

1. **Backup current implementation**
   ```bash
   cp victor_invest/handlers.py victor_invest/handlers.py.backup
   ```

2. **Create new handlers file**
   ```bash
   touch victor_invest/handlers_v2.py
   ```

3. **Verify BaseHandler import exists**
   ```python
   from victor.framework.workflows.base_handler import BaseHandler
   from victor.framework.handler_registry import handler_decorator
   ```

### Phase 2: Migrate Simple Handlers (60 minutes)

**Order**: Easiest to most complex

1. ✅ `GenerateLookbackDatesHandler` (simplest, ~40 lines)
2. ✅ `SaveRLPredictionsHandler` (simple, ~40 lines)
3. ✅ `ProcessBacktestBatchHandler` (simple, ~50 lines)
4. ✅ `AnalyzePeersHandler` (simple, ~60 lines)
5. ✅ `FetchSECDataHandler` (moderate, ~113 lines)
6. ✅ `FetchMarketDataHandler` (moderate, ~113 lines)
7. ✅ `FetchMacroDataHandler` (moderate, ~150 lines)

**Migration Template**:
```python
@handler_decorator("<handler_name>", vertical="investment", description="<description>")
@dataclass
class <HandlerName>(BaseHandler):
    """<Docstring>."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        # Extract handler signature
        # 1. Get inputs from context
        # 2. Call tools
        # 3. Return (output, tool_calls_count)

        # For errors: raise Exception or return error dict
        # For validation: raise ValueError
```

### Phase 3: Migrate Complex Handlers (90 minutes)

**Complex handlers with special logic**:

8. ⚠️ `RunFundamentalAnalysisHandler` (needs config access)
9. ⚠️ `RunTechnicalAnalysisHandler` (needs config access)
10. ⚠️ `RunMarketContextHandler` (needs config access)
11. ⚠️ `IdentifyPeersHandler` (DB access, ~200 lines)
12. ⚠️ `GenerateReportHandler` (report generation, ~210 lines)
13. ⚠️ `RunSynthesisHandler` (LLM integration, ~600 lines)

**Special Considerations**:
- Config access: Use dependency injection or pass via context
- DB access: Keep existing pattern, just change return type
- LLM calls: No change needed, just return type

### Phase 4: Update Registration (5 minutes)

**Before**:
```python
HANDLERS = {
    "fetch_sec_data": FetchSECDataHandler(),
    # ... 13 more
}

def register_handlers() -> None:
    global _handlers_registered
    if _handlers_registered:
        return
    from victor.workflows.executor import register_compute_handler
    for name, handler in HANDLERS.items():
        register_compute_handler(name, handler)
    _handlers_registered = True
```

**After**:
```python
# All handlers have @handler_decorator - auto-registered on import
# No HANDLERS dict needed
# register_handlers() becomes no-op for backward compatibility

def register_handlers() -> None:
    """No-op - handlers auto-registered via @handler_decorator."""
    pass
```

### Phase 5: Testing (30 minutes)

1. **Unit tests** (if available)
   ```bash
   pytest tests/victor_invest/test_handlers.py -v
   ```

2. **Integration tests** - Run workflows
   ```bash
   victor-invest analyze AAPL --mode quick
   victor-invest analyze AAPL --mode standard
   victor-invest analyze AAPL --mode comprehensive
   ```

3. **Verify all handlers registered**
   ```python
   from victor.framework.handler_registry import HandlerRegistry
   registry = HandlerRegistry()
   assert "fetch_sec_data" in registry.get_handlers("investment")
   ```

---

## Detailed Migration Example: FetchSECDataHandler

### Step 1: Identify Core Logic

**Current implementation lines 67-112** contain:
- Input validation: `symbol = context.get("symbol", "")`
- Tool invocation: `SECFilingTool().execute(...)`
- Output construction: `{"status": ..., "data": ...}`

### Step 2: Extract to execute() method

```python
@handler_decorator("fetch_sec_data", vertical="investment", description="Fetch SEC filing data")
@dataclass
class FetchSECDataHandler(BaseHandler):
    """Fetch SEC filing data for analysis."""

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> Tuple[Any, int]:
        # Input validation
        symbol = context.get("symbol", "")
        if not symbol:
            raise ValueError("No symbol provided")

        # Tool invocation
        from victor_invest.tools.sec_filing import SECFilingTool

        sec_tool = SECFilingTool()
        result = await sec_tool.execute(
            {},  # _exec_ctx
            symbol=symbol,
            action="get_company_facts",
        )

        # Return (output, tool_calls_count)
        return {
            "status": "success" if result.success else "error",
            "data": result.output if result.success else None,
            "error": result.error if not result.success else None,
        }, 0
```

### Step 3: Remove Boilerplate

**Removed**:
- `start_time = time.time()` (line 67)
- All `NodeResult` constructions (lines 71-76, 98-103, 107-112)
- All `try/except` blocks (lines 78-105)
- `context.set()` call (line 96)
- Manual timing in returns

**Result**: 56 lines → 36 lines (-36%)

---

## Error Handling Strategy

### Old Pattern: Return error NodeResult

```python
except Exception as e:
    logger.error(f"Error: {e}")
    return NodeResult(
        node_id=node.id,
        status=NodeStatus.FAILED,
        error=str(e),
        duration_seconds=time.time() - start_time,
    )
```

### New Pattern: Raise exception

```python
# Option 1: Let BaseHandler handle it
if not symbol:
    raise ValueError("No symbol provided")

# Option 2: Return error dict
if not symbol:
    return {
        "status": "error",
        "error": "No symbol provided"
    }, 0

# Option 3: Explicit error with logging
if not symbol:
    logger.error("No symbol provided")
    raise ValueError("No symbol provided")
```

**Recommendation**: Use Option 2 (return error dict) for validation errors, let exceptions propagate for system errors.

---

## Rollback Plan

If migration fails:

1. **Immediate rollback**:
   ```bash
   cp victor_invest/handlers.py.backup victor_invest/handlers.py
   ```

2. **Verify restore**:
   ```bash
   git diff victor_invest/handlers.py
   # Should show no changes
   ```

3. **Test rollback**:
   ```bash
   victor-invest analyze AAPL --mode quick
   ```

---

## Post-Migration Checklist

- [ ] All 14 handlers migrated to `@handler_decorator` + `BaseHandler`
- [ ] `register_handlers()` is no-op
- [ ] `HANDLERS` dict removed
- [ ] All workflow tests pass
- [ ] Manual smoke test: `victor-invest analyze AAPL --mode comprehensive`
- [ ] No warnings in logs about handler registration
- [ ] Code review completed
- [ ] Documentation updated (CLAUDE.md)
- [ ] Backup file removed (`handlers.py.backup`)

---

## Open Questions

1. **Config access in handlers**: Should we pass via context or use dependency injection?
   - **Decision**: Pass via context for now, refactor to DI later if needed

2. **LLM client cleanup**: `RunSynthesisHandler` has `finally` block for cleanup
   - **Decision**: Implement `cleanup()` method in handler if needed

3. **Helper functions**: `_format_fundamental`, `_format_technical` are module-level
   - **Decision**: Keep as-is, no migration needed

4. **Backwards compatibility**: Keep `register_handlers()` for existing code
   - **Decision**: Yes, make it a no-op

---

## Success Criteria

1. ✅ All handlers use `@handler_decorator`
2. ✅ All handlers extend `BaseHandler`
3. ✅ All handlers return `Tuple[Any, int]`
4. ✅ Zero manual `NodeResult` constructions
5. ✅ Auto-registration on import
6. ✅ All tests pass
7. ✅ Manual smoke test passes
8. ✅ Code reduced by >80%

---

## References

- Victor Framework BaseHandler: `/Users/vijaysingh/code/codingagent/victor/framework/workflows/base_handler.py`
- Handler Registry: `/Users/vijaysingh/code/codingagent/victor/framework/handler_registry.py`
- Example Migration: `/Users/vijaysingh/code/codingagent/victor/coding/handlers.py`
- Victor Docs: `/Users/vijaysingh/code/codingagent/docs/migration_v1/MIGRATION_WORKFLOWS.md`
