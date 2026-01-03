# Utils to Clean Architecture Migration Strategy

**Created**: 2025-11-13
**Status**: ACTIVE MIGRATION
**Goal**: Eliminate all imports from `src/investigator/` to `utils/` by migrating duplicates and consolidating unique code

---

## Overview

This document outlines the strategy for migrating the remaining `utils/` modules to the clean architecture in `src/investigator/`. After archiving 6 modules (cache/ + 5 standalone utilities), we have **4 high-priority duplicates** with active imports that need migration.

---

## Current State

### Archived Modules ✅

**Completed 2025-11-13**:
- `utils/cache/` → `src/investigator/infrastructure/cache/` (3 files, import shims only)
- `backup_reports.py` → archived (unused, 0 imports)
- `cache_cleanup.py` → archived (unused, functionality integrated)
- `form4_monitor.py` → archived (unused, 0 imports)
- `market_regime_visualizer.py` → archived (unused, 0 imports)
- `prompt_optimizer.py` → archived (unused, 0 imports)

### Active Duplicates (Require Import Migration)

| Utils Module | Equivalent in src/ | Active Imports | Priority |
|--------------|-------------------|----------------|----------|
| `db.py` | `infrastructure/database/db.py` | 7 | HIGH |
| `ticker_cik_mapper.py` | `infrastructure/database/ticker_mapper.py` | 7 | HIGH |
| `data_normalizer.py` | `domain/services/data_normalizer.py` | 5 | HIGH |
| `sec_data_strategy.py` | `infrastructure/sec/data_strategy.py` | 2 | MEDIUM |
| `sec_companyfacts_extractor.py` | `infrastructure/sec/companyfacts_extractor.py` | 0* | LOW |

*Note: `sec_companyfacts_extractor.py` has 0 imports from `src/investigator/` but is imported by `utils/sec_data_strategy.py` (which is imported by src/).

---

## Migration Strategy: Import Shim Pattern

For modules with active imports, we'll use the **Import Shim Pattern** to ensure backward compatibility during migration:

### Phase 1: Create Import Shim

Replace the utils module with a shim that re-exports from the new location:

```python
# utils/db.py (AFTER MIGRATION - becomes import shim)
"""
DEPRECATED: Import shim for backward compatibility.
This module has been migrated to src/investigator/infrastructure/database/

Canonical import:
    from investigator.infrastructure.database.db import get_engine, get_session

DO NOT use this shim in new code. It will be archived once all imports are updated.
"""
import warnings

# Re-export from new location
from investigator.infrastructure.database.db import (
    get_engine,
    get_session,
    init_database,
    # ... all public exports
)

warnings.warn(
    "utils.db is deprecated. Use 'from investigator.infrastructure.database import db' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['get_engine', 'get_session', 'init_database']
```

### Phase 2: Update Import Statements

Update all files in `src/investigator/` to use the new canonical import:

```python
# BEFORE (legacy import)
from utils.db import get_engine, get_session

# AFTER (canonical import)
from investigator.infrastructure.database.db import get_engine, get_session
```

### Phase 3: Verify and Test

```bash
# 1. Verify no imports bypass the shim (direct from old location)
grep -r "from utils\.MODULE_NAME" src/investigator/ --include="*.py"
# Should return 0 results

# 2. Run full test suite
pytest tests/unit/ -v

# 3. Run integration test
python3 cli_orchestrator.py analyze AAPL -m standard
```

### Phase 4: Archive the Shim

Once all imports are updated and tests pass:

```bash
git mv utils/MODULE_NAME archive/legacy-utils/MODULE_NAME
git add archive/legacy-utils/README.md
git commit -m "archive(utils): move MODULE_NAME to archive (import shim, all imports migrated)"
```

---

## Priority 1: Database Module (db.py)

**Active Imports**: 7
**Effort**: 1-2 hours
**Risk**: LOW (clean duplicate, well-tested)

### Files to Update

1. `src/investigator/domain/agents/fundamental/agent.py`
2. `src/investigator/infrastructure/cache/rdbms_cache_handler.py`
3. `src/investigator/application/synthesizer.py`
4. `src/investigator/infrastructure/sec/sec_strategies.py`
5. `src/investigator/infrastructure/sec/quarterly_processor.py`
6. `src/investigator/infrastructure/sec/data_strategy.py`
7. `src/investigator/interfaces/cli/orchestrator.py`

### Migration Steps

```bash
# 1. Compare implementations to ensure src/ version is canonical
diff utils/db.py src/investigator/infrastructure/database/db.py

# 2. If utils/ has unique code, merge into src/ version first
# (Review needed before proceeding)

# 3. Create import shim in utils/db.py
# (Replace contents with shim pattern above)

# 4. Update all 7 import statements
sed -i '' 's/from utils\.db import/from investigator.infrastructure.database.db import/g' \
  src/investigator/domain/agents/fundamental/agent.py \
  src/investigator/infrastructure/cache/rdbms_cache_handler.py \
  src/investigator/application/synthesizer.py \
  src/investigator/infrastructure/sec/sec_strategies.py \
  src/investigator/infrastructure/sec/quarterly_processor.py \
  src/investigator/infrastructure/sec/data_strategy.py \
  src/investigator/interfaces/cli/orchestrator.py

# 5. Verify no syntax errors
python3 -m py_compile src/investigator/**/*.py

# 6. Run tests
pytest tests/unit/ -v

# 7. Run integration test
python3 cli_orchestrator.py analyze AAPL -m standard

# 8. Archive the shim
git mv utils/db.py archive/legacy-utils/db.py
```

---

## Priority 2: Ticker/CIK Mapper (ticker_cik_mapper.py)

**Active Imports**: 7
**Effort**: 1-2 hours
**Risk**: LOW (clean duplicate)

### Files to Update

1. `src/investigator/domain/agents/fundamental/agent.py`
2. `src/investigator/infrastructure/cache/rdbms_cache_handler.py`
3. `src/investigator/infrastructure/sec/sec_facade.py`
4. `src/investigator/infrastructure/sec/sec_strategies.py`
5. `src/investigator/infrastructure/sec/quarterly_processor.py`
6. `src/investigator/infrastructure/sec/companyfacts_extractor.py`
7. `src/investigator/domain/agents/sec.py`

### Migration Command

```bash
sed -i '' 's/from utils\.ticker_cik_mapper import/from investigator.infrastructure.database.ticker_mapper import/g' \
  [list of 7 files above]
```

---

## Priority 3: Data Normalizer (data_normalizer.py)

**Active Imports**: 5
**Effort**: 1 hour
**Risk**: LOW

### Files to Update

1. `src/investigator/domain/agents/fundamental/agent.py`
2. `src/investigator/domain/agents/technical.py`
3. `src/investigator/domain/agents/synthesis.py`
4. `src/investigator/infrastructure/logging_utils.py`
5. `src/investigator/domain/agents/base.py`

### Migration Command

```bash
sed -i '' 's/from utils\.data_normalizer import/from investigator.domain.services.data_normalizer import/g' \
  [list of 5 files above]
```

---

## Priority 4: SEC Data Strategy (sec_data_strategy.py)

**Active Imports**: 2
**Effort**: 30 minutes
**Risk**: LOW

### Files to Update

1. `src/investigator/domain/agents/fundamental/agent.py`
2. `src/investigator/infrastructure/sec/companyfacts_extractor.py`

**Note**: After migrating this, `utils/sec_companyfacts_extractor.py` will have 0 imports and can be archived immediately.

---

## Unique Utils Modules (No src/ Equivalent)

These modules have NO equivalent in `src/investigator/` and are actively used. They need to be **migrated** (not shimmed):

### Priority 1: Critical Valuation Infrastructure

**Estimated Effort**: 3-5 days

1. **`quarterly_calculator.py`** (1361 lines)
   - **Target**: `src/investigator/domain/services/quarterly_calculator.py`
   - **Blockers**: Contains critical Q4/fiscal year fixes
   - **Referenced By**: `utils/dcf_valuation.py`
   - **Strategy**: Migrate first, then update dcf_valuation.py imports

2. **`dcf_valuation.py`** (2369 lines)
   - **Target**: `src/investigator/domain/services/dcf_calculator.py`
   - **Blockers**: Depends on quarterly_calculator.py
   - **Referenced By**: `utils/valuation/models/dcf_model.py`, fundamental agent
   - **Strategy**: Migrate after quarterly_calculator.py

3. **`gordon_growth_model.py`** (564 lines)
   - **Target**: `src/investigator/domain/services/ggm_calculator.py`
   - **Blockers**: None
   - **Referenced By**: `utils/valuation/models/ggm_model.py`
   - **Strategy**: Can migrate independently

### Priority 2: Infrastructure

4. **`monitoring.py`** (697 lines)
   - **Target**: `src/investigator/infrastructure/monitoring/metrics.py`
   - **Effort**: 1-2 days

5. **`event_bus.py`** (unknown size)
   - **Target**: `src/investigator/infrastructure/events/event_bus.py`
   - **Effort**: 1 day

6. **`market_data_fetcher.py`** (428 lines)
   - **Target**: `src/investigator/infrastructure/database/market_data_fetcher.py`
   - **Effort**: 1 day

---

## Migration Timeline

### Week 1: Import Shim Migration (4 duplicates)

**Day 1-2**: Database and Ticker/CIK Mapper
- Migrate `utils/db.py` → shim → archive
- Migrate `utils/ticker_cik_mapper.py` → shim → archive
- Run full test suite

**Day 3**: Data Normalizer and SEC Data Strategy
- Migrate `utils/data_normalizer.py` → shim → archive
- Migrate `utils/sec_data_strategy.py` → shim → archive
- Archive `utils/sec_companyfacts_extractor.py` (becomes unused)

**Day 4-5**: Validation and Documentation
- Run comprehensive integration tests (AAPL, META, ZS, DASH)
- Update CLAUDE.md with migration completion
- Document any edge cases encountered

### Week 2-3: Critical Valuation Migration

**Week 2**: Quarterly Calculator + DCF
- Migrate `quarterly_calculator.py` (3 days)
- Migrate `dcf_valuation.py` (2 days)

**Week 3**: GGM + Testing
- Migrate `gordon_growth_model.py` (1 day)
- Comprehensive valuation testing (2 days)
- Update fundamental agent to use new services (2 days)

### Week 4+: Infrastructure Migration (Lower Priority)

- Monitoring, event_bus, market_data_fetcher
- Can be done incrementally as time permits

---

## Verification Checklist

After each migration:

- [ ] All imports updated to canonical path
- [ ] Import shim created (for duplicates)
- [ ] Unit tests pass: `pytest tests/unit/ -v`
- [ ] Integration test passes: `python3 cli_orchestrator.py analyze AAPL -m standard`
- [ ] No imports from old location: `grep -r "from utils.MODULE" src/investigator/ --include="*.py"`
- [ ] Archive README updated
- [ ] Commit created with proper message
- [ ] Git history preserved (using `git mv`)

---

## Success Criteria

**Phase 1 Complete** (Import Shim Migration):
- [ ] 0 imports from `src/investigator/` to `utils/db.py`
- [ ] 0 imports from `src/investigator/` to `utils/ticker_cik_mapper.py`
- [ ] 0 imports from `src/investigator/` to `utils/data_normalizer.py`
- [ ] 0 imports from `src/investigator/` to `utils/sec_data_strategy.py`
- [ ] All 4 modules archived with import shims
- [ ] All tests pass
- [ ] Full analysis pipeline works (tested with 3+ symbols)

**Phase 2 Complete** (Valuation Migration):
- [ ] `quarterly_calculator.py` migrated to `domain/services/`
- [ ] `dcf_valuation.py` migrated to `domain/services/`
- [ ] `gordon_growth_model.py` migrated to `domain/services/`
- [ ] Fundamental agent updated to use new services
- [ ] Valuation accuracy validated (compare old vs new outputs)

**Final Success**:
- [ ] Zero imports from `src/investigator/` to `utils/` (excluding allowed modules)
- [ ] Allowed utils modules: `canonical_key_mapper.py`, `financial_calculators.py`, `json_utils.py`, `period_utils.py`
- [ ] Clean architecture fully enforced
- [ ] Documentation complete and up-to-date

---

## Rollback Plan

If issues arise during migration:

```bash
# 1. Identify the problematic module
MODULE_NAME="db"

# 2. Restore from archive
git mv archive/legacy-utils/${MODULE_NAME}.py utils/${MODULE_NAME}.py

# 3. Revert import changes
git checkout HEAD~1 -- src/investigator/  # Revert to previous commit

# 4. Document the issue
echo "ROLLBACK: ${MODULE_NAME} - [reason]" >> docs/MIGRATION_ROLLBACKS.md

# 5. Investigate and fix
# (Address the root cause before re-attempting migration)
```

---

## Notes

- **Import Shim Pattern** provides zero-downtime migration with immediate rollback capability
- **Verification at each step** ensures stability throughout the process
- **Git history preserved** using `git mv` instead of delete/create
- **Deprecation warnings** alert developers to update their imports
- **Archive directory** maintains rollback safety indefinitely

---

**Last Updated**: 2025-11-13
**Next Review**: After Phase 1 completion
