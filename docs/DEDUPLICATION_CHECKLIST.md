# InvestiGator Deduplication Checklist & Action Items

## Files Created by This Analysis

1. **ARCHITECTURE_DUPLICATION_ANALYSIS.md** - Comprehensive analysis
2. **ARCHITECTURE_DUPLICATION_VISUAL.txt** - Visual dependency map
3. **DEDUPLICATION_CHECKLIST.md** - This file

---

## IMMEDIATE FIXES REQUIRED (BLOCKING)

### Step 1: Fix Domain Agents - Stop Importing from Old Base
**Problem**: Domain agents import from old `agents.base` instead of new domain models

**Files to Fix** (5 files):
```
src/investigator/domain/agents/fundamental.py:   Line 17
src/investigator/domain/agents/market_context.py: Line 17
src/investigator/domain/agents/technical.py:      Line 19
src/investigator/domain/agents/synthesis.py:      Line 18
src/investigator/domain/agents/sec.py:            Line 18
```

**Current Import** (WRONG):
```python
from agents.base import AgentResult, AgentTask, AnalysisType, InvestmentAgent, TaskStatus
```

**Should Be** (CORRECT):
```python
from investigator.domain.models.analysis import AgentResult, AgentTask, AnalysisType, TaskStatus
from investigator.domain.agents.base import InvestmentAgent
```

**Action Items**:
- [ ] Update fundamental.py - line 17
- [ ] Update market_context.py - line 17  
- [ ] Update technical.py - line 19
- [ ] Update synthesis.py - line 18
- [ ] Update sec.py - line 18
- [ ] Run tests to verify no breakage
- [ ] Commit: "fix(domain): update agents to import from domain models, not legacy agents/"

---

### Step 2: Consolidate Model Definitions
**Problem**: AgentTask, AgentResult, TaskStatus defined in both locations

**Files Involved**:
```
agents/base.py (lines 50-300)              ← DELETE these definitions
src/investigator/domain/models/analysis.py ← USE ONLY this location
```

**What to Do**:
1. [ ] Verify all definitions in domain/models/analysis.py are complete
2. [ ] Check that OLD agents/base.py definitions are identical
3. [ ] Keep ONLY domain/models/analysis.py definitions
4. [ ] Delete duplicate definitions from agents/base.py
5. [ ] Test agent instantiation still works
6. [ ] Commit: "refactor(models): consolidate dataclass definitions to domain layer"

---

### Step 3: Fix Domain Agents Cache Imports
**Problem**: Domain agents still import cache from old `utils.cache`

**Files to Fix** (5 files):
```
src/investigator/domain/agents/fundamental.py
src/investigator/domain/agents/market_context.py
src/investigator/domain/agents/technical.py
src/investigator/domain/agents/synthesis.py
src/investigator/domain/agents/sec.py
```

**Current Import** (WRONG):
```python
from utils.cache.cache_manager import CacheManager
from utils.cache.cache_types import CacheType
```

**Should Be** (CORRECT):
```python
from investigator.infrastructure.cache import CacheManager, CacheType
```

**Action Items**:
- [ ] Find all cache imports in each agent file
- [ ] Replace with new architecture imports
- [ ] Run domain agent tests to verify
- [ ] Commit: "refactor(cache): update domain agents to use infrastructure cache layer"

---

## SHORT-TERM FIXES (This Week)

### Step 4: Update Test Imports
**Problem**: 80+ test files use old agent imports

**Files to Update**:
```
tests/test_*.py (80+ files)
tests/agents/*.py
tests/integration/test_*.py
```

**Current Pattern** (WRONG):
```python
from agents.base import AgentTask, TaskStatus, Priority
from agents.fundamental_agent import FundamentalAnalysisAgent
```

**Should Be** (CORRECT):
```python
from investigator.domain.models.analysis import AgentTask, TaskStatus, Priority
from investigator.domain.agents.fundamental import FundamentalAnalysisAgent
```

**Script to Find All Occurrences**:
```bash
# Find all test files using old imports
grep -r "from agents" tests/ --include="*.py" | cut -d: -f1 | sort -u

# Should output ~20 unique test files
```

**Action Items**:
- [ ] Create script to update test imports in bulk
- [ ] Run: `grep -r "from agents\." tests/ --include="*.py" > /tmp/test_files_to_update.txt`
- [ ] Update each file or use sed/awk to bulk replace
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Verify 100% tests still pass
- [ ] Commit: "test(refactor): update imports to use new domain layer"

---

### Step 5: Consolidate Cache Layer
**Problem**: Cache layer duplicated in two locations (100% identical)

**Old Location** (TO DELETE):
```
utils/cache/
├─ cache_manager.py
├─ cache_types.py
├─ cache_base.py
├─ file_cache_handler.py
├─ rdbms_cache_handler.py
├─ parquet_cache_handler.py
├─ cache_cleaner.py
├─ cache_inspector.py
├─ cache_key_builder.py
├─ market_regime_cache.py
└─ sec_cache_analyzer.py
```

**New Location** (TO KEEP):
```
src/investigator/infrastructure/cache/
└─ [all same files]
```

**Verification Script**:
```bash
# Verify both locations have identical files
for file in cache_manager.py cache_types.py cache_base.py; do
  diff -q utils/cache/$file src/investigator/infrastructure/cache/$file
done

# List all imports of old cache location
grep -r "from utils.cache" . --include="*.py" | grep -v "src/" | grep -v "test" | wc -l
```

**Action Items**:
- [ ] Run verification to confirm 100% duplication
- [ ] Find all non-test imports of `utils.cache`
- [ ] Update those imports to `investigator.infrastructure.cache`
- [ ] Delete `/utils/cache/` directory
- [ ] Create `/utils/cache/__init__.py` with re-exports for backward compatibility (optional)
- [ ] Run tests: `pytest tests/ -v`
- [ ] Commit: "refactor(cache): consolidate to single location, remove duplicate layer"

---

## MEDIUM-TERM FIXES (Next Week)

### Step 6: Remove Old Agents Directory
**Problem**: `/agents/` directory still exists but should be deleted

**Files to Delete**:
```
agents/__init__.py
agents/base.py (after consolidating models)
agents/orchestrator.py (now in application layer)
agents/fundamental_agent.py (now in domain/agents)
agents/synthesis_agent.py (now in domain/agents)
agents/technical_agent.py (now in domain/agents)
agents/sec_agent.py (now in domain/agents)
agents/etf_market_context_agent.py (now in domain/agents)
agents/manager.py (check if in new architecture)
agents/peer_group_orchestrator.py (check if still needed)
```

**Pre-Deletion Checks**:
```bash
# Verify nothing imports from agents/ outside of tests
grep -r "from agents import\|from agents\." . --include="*.py" | grep -v "src/" | grep -v test | grep -v __pycache__

# Should return ZERO results before deletion
```

**Action Items**:
- [ ] Run pre-deletion check script
- [ ] Confirm ZERO non-test imports of old agents/
- [ ] Delete entire `/agents/` directory
- [ ] Run tests: `pytest tests/ -v`
- [ ] Run CLI: `python3 cli_orchestrator.py analyze AAPL -m standard`
- [ ] Commit: "refactor(cleanup): remove legacy agents directory after full migration"

---

### Step 7: Consolidate Remaining Utilities
**Problem**: Various utilities still in `/utils/` not yet in infrastructure layer

**Files to Migrate**:
```
utils/data_normalizer.py → infrastructure/services/data_normalizer.py
utils/sec_data_processor.py → infrastructure/sec/data_processor.py
utils/sec_quarterly_processor.py → infrastructure/sec/quarterly_processor.py
utils/db.py → infrastructure/database/db.py (check if exists)
utils/market_data_fetcher.py → infrastructure/data/market_fetcher.py
utils/monitoring.py → infrastructure/monitoring/metrics.py
utils/event_bus.py → infrastructure/event_bus.py
```

**Action Items**:
- [ ] Audit `/utils/` directory contents
- [ ] Map each file to infrastructure layer location
- [ ] Create move plan
- [ ] Execute moves incrementally
- [ ] Update all imports throughout codebase
- [ ] Keep `/utils/` directory with re-export shims for backward compatibility
- [ ] Commit: "refactor(infrastructure): consolidate utilities into infrastructure layer"

---

## VERIFICATION CHECKLIST

### After Completing Step 1 (Domain agents fix):
- [ ] `pytest tests/unit/domain/agents/ -v` passes
- [ ] `pytest tests/integration/ -v` passes  
- [ ] No import errors when loading investigator.application module
- [ ] `python3 cli_orchestrator.py analyze AAPL -m quick` completes successfully

### After Completing Step 5 (Cache consolidation):
- [ ] No files in `/utils/cache/` anymore (or only __init__.py shim)
- [ ] `pytest tests/cache/ -v` passes
- [ ] `python3 cli_orchestrator.py analyze AAPL -m standard` uses new cache layer
- [ ] Cache hit/miss statistics still work

### After Completing Step 6 (Remove agents/):
- [ ] `/agents/` directory deleted
- [ ] All tests pass: `pytest tests/ -v --cov=src/`
- [ ] `python3 cli_orchestrator.py batch AAPL MSFT GOOGL -m standard` works
- [ ] Full analysis runs without warnings about import paths

### After All Steps Complete:
- [ ] No duplication in codebase: `cloc --exclude-dir=.git,.pytest_cache src/ | grep -A5 "Language"`
- [ ] Clear separation of concerns
- [ ] Single import path for each component
- [ ] Clean architecture verified: `grep -r "from agents\." . --include="*.py"` returns ZERO results (except comments)

---

## ESTIMATES

| Phase | Task | Effort | Risk | Priority |
|-------|------|--------|------|----------|
| 1 | Fix domain agent imports | 2h | Very Low | **CRITICAL** |
| 2 | Consolidate models | 1h | Very Low | **CRITICAL** |
| 3 | Fix cache imports | 1h | Very Low | **CRITICAL** |
| 4 | Update test imports | 4h | Low | **HIGH** |
| 5 | Remove cache duplication | 2h | Low | **HIGH** |
| 6 | Remove agents/ directory | 1h | Medium | **MEDIUM** |
| 7 | Consolidate utilities | 6h | Medium | **MEDIUM** |
| | **TOTAL** | **17h** | | |

---

## GIT WORKFLOW

### Recommended Commit Sequence:

1. `fix(domain): update agents to import from domain models`
   - Subject of immediate fix from Step 1

2. `refactor(models): consolidate dataclass definitions`
   - Step 2

3. `refactor(cache): update domain agents to use infrastructure layer`
   - Step 3

4. `test(refactor): update imports to use new domain layer`
   - Step 4

5. `refactor(cache): consolidate cache layer, remove utils/cache duplication`
   - Step 5

6. `refactor(cleanup): remove legacy agents directory`
   - Step 6

7. `refactor(infrastructure): consolidate utilities migration`
   - Step 7

Each commit should be accompanied by test runs to ensure no regressions.

---

## SUCCESS CRITERIA

When deduplication is complete:

- [ ] Zero lines of duplicated code between old/new architecture
- [ ] Single import path for every class (no ambiguity)
- [ ] All 80+ tests pass using NEW architecture imports
- [ ] No warnings about import deprecation
- [ ] CLI commands work identically
- [ ] Performance metrics unchanged
- [ ] Code coverage maintained or improved
- [ ] Architecture is now a true clean architecture (no circular dependencies)

---

## ROLLBACK PLAN

If anything breaks during deduplication:

1. [ ] Revert last commit: `git revert HEAD`
2. [ ] Identify what broke: check test failures
3. [ ] Fix in isolation: create focused bugfix branch
4. [ ] Test thoroughly: run `pytest -v` before pushing
5. [ ] Re-apply: cherry-pick fixed commit

The test suite provides comprehensive safety net for this refactoring.

