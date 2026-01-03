# Legacy Utils Archive

**Purpose**: This directory contains utils/ modules that have been **fully migrated** to the new clean architecture in `src/investigator/`. These files are archived for reference but should NOT be used in active code.

**Migration Date**: 2025-11-13
**Status**: ARCHIVED - Do not import from these modules

---

## Archived Modules

### ✅ cache/ (Migrated: 2025-11-13)

**Original Location**: `utils/cache/`
**New Location**: `src/investigator/infrastructure/cache/`
**Migration Status**: COMPLETE - All functionality moved

**Files Archived**:
- `cache_manager.py` (26 lines) - Import shim only
- `cache_types.py` (12 lines) - Import shim only
- `__init__.py` (3 lines) - Package init

**Canonical Implementation**:
```python
# DO NOT USE:
from utils.cache.cache_manager import CacheManager  # ❌ ARCHIVED

# USE INSTEAD:
from investigator.infrastructure.cache import CacheManager  # ✅ CORRECT
```

**Verification**: No active imports from `utils.cache` exist in `src/investigator/` as of archival date.

---

### ✅ Standalone Utilities (Archived: 2025-11-13)

**Status**: UNUSED - Zero imports from src/investigator/ or active code

**Files Archived**:
- `backup_reports.py` (6.2 KB) - Report backup utility, unused
- `cache_cleanup.py` (12 KB) - Cache cleanup utility, functionality integrated into infrastructure
- `form4_monitor.py` (19 KB) - Insider trading monitor, unused
- `market_regime_visualizer.py` (26 KB) - Market visualization utility, unused
- `prompt_optimizer.py` (9.4 KB) - LLM context optimization, unused

**Reason for Archival**: These standalone utilities were never integrated into the active codebase. Comprehensive search found 0 imports from `src/investigator/` to any of these modules.

**Verification**:
```bash
grep -r "cache_cleanup\|backup_reports\|form4_monitor\|prompt_optimizer\|market_regime_visualizer" \
  src/investigator/ --include="*.py" | wc -l
# Returns: 0
```

---

## Why Archive Instead of Delete?

1. **Historical Reference**: Preserves git history and allows comparison if issues arise
2. **Rollback Safety**: Can quickly restore if migration has unforeseen issues
3. **Documentation**: Shows what was migrated and when
4. **Debugging**: Helps understand code evolution during troubleshooting

---

## When to Archive a Module

Archive a utils/ module ONLY after:

1. ✅ **Complete migration** - All functionality moved to `src/investigator/`
2. ✅ **Import verification** - No active imports from archived module:
   ```bash
   grep -r "from utils.MODULE_NAME" src/investigator/ --include="*.py"
   # Should return 0 results
   ```
3. ✅ **Test validation** - All tests pass with new imports
4. ✅ **Integration test** - Full analysis pipeline works
5. ✅ **Documentation** - Migration documented in this README

---

## Archival Process

```bash
# 1. Verify no active imports
grep -r "from utils.MODULE_NAME" src/investigator/ --include="*.py"
grep -r "import utils.MODULE_NAME" src/investigator/ --include="*.py"

# 2. Move to archive
git mv utils/MODULE_NAME archive/legacy-utils/MODULE_NAME

# 3. Update this README with migration details

# 4. Commit archival
git add archive/legacy-utils/README.md
git commit -m "archive(utils): move MODULE_NAME to archive (migrated to src/investigator/PACKAGE)"

# 5. Verify system still works
python3 cli_orchestrator.py test-system
pytest tests/unit/ -v
```

---

## Pending Migrations

These modules have **NOT** been archived yet (migration in progress or planned):

### Priority 1: CRITICAL (Active Migration)
- `quarterly_calculator.py` - Being migrated to `domain/services/`
- `dcf_valuation.py` - Being migrated to `domain/services/`
- `gordon_growth_model.py` - Being migrated to `domain/services/`
- `valuation/orchestrator.py` - Architecture decision pending

### Priority 2: Infrastructure
- `db.py` - Near duplicate, consolidation needed
- `monitoring.py` - To be migrated to `infrastructure/monitoring/`
- `event_bus.py` - To be migrated to `infrastructure/events/`
- `market_data_fetcher.py` - To be migrated to `infrastructure/database/`

### Priority 3: Consolidation Needed
- `sec_companyfacts_extractor.py` - Near duplicate with `src/.../sec/companyfacts_extractor.py`
- `ticker_cik_mapper.py` - Near duplicate with `src/.../database/ticker_mapper.py`
- `data_normalizer.py` - Near duplicate with `src/.../services/data_normalizer.py`

### Keep in Utils (For Now)
- `canonical_key_mapper.py` - 247 XBRL mappings (consider moving to `domain/value_objects/`)
- `financial_calculators.py` - Helper functions
- `json_utils.py` - Utilities
- `period_utils.py` - Period helpers

---

## Rollback Procedure (If Needed)

If archived code needs to be restored:

```bash
# 1. Restore from archive
git mv archive/legacy-utils/MODULE_NAME utils/MODULE_NAME

# 2. Revert imports in affected files
# (Use git history to find what changed)

# 3. Test thoroughly
pytest tests/ -v

# 4. Document why rollback was needed
echo "ROLLBACK REASON: [explain issue]" >> archive/ROLLBACK_LOG.md
```

---

### ✅ db.py (Migrated: 2025-11-13)

**Original Location**: `utils/db.py`
**New Location**: `src/investigator/infrastructure/database/db.py`
**Migration Status**: COMPLETE - All imports updated, import shim archived

**Migration Method**: Import Shim Pattern (Phase 1, Priority 1)
**Active Imports Updated**: 7 files

**Files Updated**:
- `src/investigator/domain/agents/fundamental/agent.py` (5 imports)
- `src/investigator/infrastructure/cache/rdbms_cache_handler.py`
- `src/investigator/application/synthesizer.py`
- `src/investigator/sec/sec_strategies.py`
- `src/investigator/infrastructure/sec/quarterly_processor.py`
- `src/investigator/infrastructure/sec/data_strategy.py`
- `src/investigator/cli/orchestrator.py`

**Canonical Implementation**:
```python
# DO NOT USE:
from utils.db import get_db_manager  # ❌ ARCHIVED

# USE INSTEAD:
from investigator.infrastructure.database.db import get_db_manager  # ✅ CORRECT
```

**Verification**: 0 active imports from `utils.db` exist in `src/investigator/` as of archival date.

---

### ✅ ticker_cik_mapper.py (Migrated: 2025-11-13)

**Original Location**: `utils/ticker_cik_mapper.py`
**New Location**: `src/investigator/infrastructure/database/ticker_mapper.py`
**Migration Status**: COMPLETE - All imports updated, import shim archived

**Migration Method**: Import Shim Pattern (Phase 1, Priority 2)
**Active Imports Updated**: 5 files (6 total imports)

**Files Updated**:
- `src/investigator/sec/sec_facade.py`
- `src/investigator/sec/sec_strategies.py`
- `src/investigator/infrastructure/sec/companyfacts_extractor.py` (2 imports)
- `src/investigator/infrastructure/sec/quarterly_processor.py`
- `src/investigator/domain/agents/sec.py`

**Canonical Implementation**:
```python
# DO NOT USE:
from utils.ticker_cik_mapper import TickerCIKMapper  # ❌ ARCHIVED

# USE INSTEAD:
from investigator.infrastructure.database.ticker_mapper import TickerCIKMapper  # ✅ CORRECT
```

**Verification**: 0 active imports from `utils.ticker_cik_mapper` exist in `src/investigator/` as of archival date.

---

### ✅ data_normalizer.py (Migrated: 2025-11-13)

**Original Location**: `utils/data_normalizer.py`
**New Location**: `src/investigator/domain/services/data_normalizer.py`
**Migration Status**: COMPLETE - All imports updated, critical bug fixed, import shim archived

**Migration Method**: Import Shim Pattern + Bug Fix (Phase 1, Priority 3)
**Active Imports Updated**: 5 files (9 total imports)
**Critical Bug Fixed**: `FIELD_NAME_MAP` → `FIELD_MAPPINGS` in assess_completeness()

**Files Updated**:
- `src/investigator/domain/agents/technical.py`
- `src/investigator/domain/agents/fundamental/agent.py` (3 imports)
- `src/investigator/domain/agents/fundamental/logging_utils.py`
- `src/investigator/domain/agents/base.py` (2 imports)
- `src/investigator/domain/agents/synthesis.py` (4 imports)

**Canonical Implementation**:
```python
# DO NOT USE:
from utils.data_normalizer import DataNormalizer  # ❌ ARCHIVED

# USE INSTEAD:
from investigator.domain.services.data_normalizer import DataNormalizer  # ✅ CORRECT
```

**Verification**: 0 active imports from `utils.data_normalizer` exist in `src/investigator/` as of archival date.

---

### ✅ sec_data_strategy.py (Migrated: 2025-11-13)

**Original Location**: `utils/sec_data_strategy.py`
**New Location**: `src/investigator/infrastructure/sec/data_strategy.py`
**Migration Status**: COMPLETE - All imports updated, Q1 fiscal year fix added to canonical version

**Migration Method**: Import Shim Pattern + Q1 Fix (Phase 1, Priority 4)
**Active Imports Updated**: 2 files (4 total imports)
**Critical Enhancement**: Added Q1 fiscal year adjustment logic to canonical version (70+ lines)

**Files Updated**:
- `src/investigator/infrastructure/sec/companyfacts_extractor.py` (2 imports)
- `src/investigator/domain/agents/fundamental/agent.py` (2 imports)

**Q1 Fiscal Year Fix Added**:
The canonical version was missing critical Q1 fiscal year adjustment logic that handles:
- Non-calendar fiscal year companies (e.g., fiscal year ending Sep 30, Jan 31)
- Q1 periods that cross calendar year boundaries
- Proper fiscal year assignment based on period_end vs fiscal_year_end comparison

This fix was added to both `get_multiple_quarters()` and `get_complete_fiscal_year()` methods before migration.

**Canonical Implementation**:
```python
# DO NOT USE:
from utils.sec_data_strategy import SECDataStrategy  # ❌ ARCHIVED

# USE INSTEAD:
from investigator.infrastructure.sec.data_strategy import SECDataStrategy  # ✅ CORRECT
```

**Verification**: 0 active imports from `utils.sec_data_strategy` exist in `src/investigator/` as of archival date.

---

### ✅ sec_companyfacts_extractor.py (Archived: 2025-11-13)

**Original Location**: `utils/sec_companyfacts_extractor.py`
**New Location**: `src/investigator/infrastructure/sec/companyfacts_extractor.py` (already canonical)
**Migration Status**: COMPLETE - Dead code removal (0 active imports)

**Archival Method**: Direct archival (no import shim needed)
**Active Imports**: 0 (utils/ version was completely unused)

**Reason for Archival**:
The canonical version in `src/investigator/infrastructure/sec/companyfacts_extractor.py` was already being used exclusively. The utils/ version (55KB, 1,362 lines) was dead code with zero imports from any active code.

**Canonical Implementation** (already in use):
```python
# CORRECT (already in use everywhere):
from investigator.infrastructure.sec.companyfacts_extractor import SECCompanyFactsExtractor

# NEVER USED (archived):
from utils.sec_companyfacts_extractor import SECCompanyFactsExtractor  # ❌
```

**Verification**: 0 active imports from `utils.sec_companyfacts_extractor` exist in entire codebase as of archival date.

---

### ✅ market_data_fetcher.py (Migrated: 2025-11-13)

**Original Location**: `utils/market_data_fetcher.py`
**New Location**: `src/investigator/infrastructure/database/market_data.py`
**Migration Status**: COMPLETE - All imports updated, canonical version has enhanced features

**Migration Method**: Import Shim Pattern (Phase 1, Priority 6)
**Active Imports Updated**: 7 files (8 total imports)

**Files Updated**:
- `src/investigator/application/synthesizer.py`
- `src/investigator/infrastructure/cache/market_regime_cache.py`
- `src/investigator/domain/agents/market_context.py`
- `src/investigator/domain/agents/technical.py`
- `src/investigator/domain/agents/fundamental/agent.py` (2 imports)
- `src/investigator/domain/agents/sec.py`

**Enhancement**: The canonical version (452 lines) includes additional helper methods for metadata handling:
- `_get_symbol_metadata()` - Enhanced symbol metadata retrieval
- `_extract_int()` - Integer extraction helper
- `_format_cik()` - CIK formatting helper
- `_extract_first_nonempty()` - Helper to extract first non-empty value
- `_infer_is_etf()` - ETF detection logic
- `coerce_bool()` - Boolean coercion helper

These enhancements improve data quality and provide better handling of edge cases.

**Canonical Implementation**:
```python
# DO NOT USE:
from utils.market_data_fetcher import DatabaseMarketDataFetcher  # ❌ ARCHIVED

# USE INSTEAD:
from investigator.infrastructure.database.market_data import DatabaseMarketDataFetcher  # ✅ CORRECT
```

**Verification**: 0 active imports from `utils.market_data_fetcher` exist in `src/investigator/` as of archival date.

---

## Archive History

| Module | Archived Date | Migrated To | Reason |
|--------|---------------|-------------|--------|
| `cache/` | 2025-11-13 | `infrastructure/cache/` | Complete migration, only import shims remained |
| `backup_reports.py` | 2025-11-13 | N/A | Unused standalone utility (0 active imports) |
| `cache_cleanup.py` | 2025-11-13 | `infrastructure/cache/` (integrated) | Unused, functionality integrated into cache infrastructure |
| `form4_monitor.py` | 2025-11-13 | N/A | Unused insider trading monitor (0 active imports) |
| `market_regime_visualizer.py` | 2025-11-13 | N/A | Unused visualization utility (0 active imports) |
| `prompt_optimizer.py` | 2025-11-13 | N/A | Unused LLM context optimizer (0 active imports) |
| `db.py` | 2025-11-13 | `infrastructure/database/db.py` | Complete migration via import shim (7 imports updated) |
| `ticker_cik_mapper.py` | 2025-11-13 | `infrastructure/database/ticker_mapper.py` | Complete migration via import shim (5 files, 6 imports updated) |
| `data_normalizer.py` | 2025-11-13 | `domain/services/data_normalizer.py` | Complete migration + bug fix (5 files, 9 imports updated) |
| `sec_data_strategy.py` | 2025-11-13 | `infrastructure/sec/data_strategy.py` | Complete migration + Q1 fiscal year fix (2 files, 4 imports updated) |
| `sec_companyfacts_extractor.py` | 2025-11-13 | `infrastructure/sec/companyfacts_extractor.py` | Dead code removal - 0 active imports (55KB, 1,362 lines archived) |
| `market_data_fetcher.py` | 2025-11-13 | `infrastructure/database/market_data.py` | Complete migration via import shim (7 files, 8 imports updated, canonical version enhanced) |

---

## Related Documentation

- **Migration Guide**: `docs/MIGRATION_STRATEGY.md` (if exists)
- **Duplication Analysis**: See comprehensive analysis from 2025-11-13 session
- **Clean Architecture**: `README.adoc` - Architecture overview
- **Import Guidelines**: `docs/CLAUDE.md` - Prefer `investigator.*` imports

---

**Last Updated**: 2025-11-13
**Maintainer**: InvestiGator Development Team
