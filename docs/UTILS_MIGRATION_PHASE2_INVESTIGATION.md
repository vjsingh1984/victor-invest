# Utils Migration - Phase 2 Investigation

**Date**: 2025-11-13
**Status**: Investigation Complete - Architectural Decisions Needed
**Phase 1 Completed**: 7 modules successfully migrated using import shim pattern

---

## Investigation Summary

After completing Phase 1 migrations (7 modules with existing canonical versions), I investigated the remaining utils/ modules to determine migration paths. The findings reveal that remaining modules fall into three categories requiring different approaches:

1. **Valuation Infrastructure** - Requires architectural decision about parallel architectures
2. **Utility Modules Without Canonical Versions** - Requires module creation first
3. **Deferred Complex Migrations** - Requires full module restructuring

---

## Category 1: Valuation Infrastructure (Architectural Decision Needed)

### Discovery: Parallel Valuation Architectures

The codebase has **TWO separate valuation architectures**:

**Legacy Architecture** (`utils/`):
- `utils/dcf_valuation.py` (2,369 lines) - Imported by synthesizer.py
- `utils/gordon_growth_model.py` (564 lines) - NOT imported by src/
- `utils/quarterly_calculator.py` (1,361 lines) - Dependency for above
- `utils/valuation/` directory (12 files) - Separate older architecture, 0 imports from src/

**New Clean Architecture** (`src/investigator/domain/services/`):
- `valuation.py` - Gordon Growth Model (already migrated!)
- `parallel_valuation_orchestrator.py` - New parallel orchestrator
- `valuation_framework_planner.py` - New planner
- `fcf_growth_calculator.py` - FCF growth calculations
- `terminal_growth_calculator.py` - Terminal growth calculations

### Module Details

#### dcf_valuation.py
- **Size**: 2,369 lines
- **Imports from src/**: 1 (synthesizer.py)
- **Dependencies**: quarterly_calculator.py
- **Status**: ACTIVELY USED for DCF valuation
- **Decision Needed**: Migrate to new parallel architecture or keep legacy code?
- **Complexity**: High - actively used, large codebase

#### gordon_growth_model.py
- **Size**: 564 lines
- **Imports from src/**: 0 (only imported by unused utils/valuation/)
- **Dependencies**: quarterly_calculator.py
- **Status**: EFFECTIVELY DEAD CODE
- **Notes**: Gordon Growth Model ALREADY MIGRATED to `src/investigator/domain/services/valuation.py`
- **Decision**: Can be archived once utils/valuation/ directory status confirmed

#### quarterly_calculator.py
- **Size**: 1,361 lines
- **Imports from src/**: 0
- **Used By**: dcf_valuation.py, gordon_growth_model.py (both in utils/)
- **Status**: Dependency for valuation modules
- **Decision**: Part of overall valuation infrastructure decision

#### utils/valuation/ Directory
- **Contents**: 12 files including orchestrator.py, models/dcf_model.py, models/ggm_model.py, etc.
- **Imports from src/**: 0 (completely unused by clean architecture)
- **Status**: Separate older valuation architecture
- **Decision**: Assess whether entire directory can be archived

### Recommended Action

**REQUIRES ARCHITECTURAL DECISION**:
1. Is the new parallel valuation architecture intended to replace legacy utils/dcf_valuation.py?
2. Should synthesizer.py be updated to use the new architecture?
3. Can the entire utils/valuation/ directory be archived?
4. Once direction is determined, can proceed with migration or archival

---

## Category 2: Modules Without Canonical Versions (Require Creation)

These modules are actively imported by src/investigator/ but have NO canonical versions in the clean architecture. They would require **creating new modules first**, not just import shims.

### canonical_key_mapper.py
- **Size**: 837 lines
- **Imports from src/**: 2 files
  - `src/investigator/infrastructure/sec/data_processor.py`
  - `src/investigator/domain/agents/fundamental/agent.py` (with TODO: "Move to infrastructure")
- **Contents**: 247 XBRL tag mappings for normalization
- **Migration Pattern**: CREATE + IMPORT SHIM (not simple shim)
- **Target Location**: Undecided
  - Option 1: `src/investigator/domain/value_objects/canonical_mapper.py`
  - Option 2: `src/investigator/infrastructure/sec/canonical_mapper.py`
- **Complexity**: Medium - clear purpose, but large mapping table

### json_utils.py
- **Size**: 89 lines
- **Imports from src/**: 3 (all in sec/)
  - `src/investigator/sec/sec_adapters.py` - `safe_json_dumps()`
  - `src/investigator/sec/sec_facade.py` (2x) - `extract_json_from_text()`
- **Contents**: JSON parsing utilities
- **Migration Pattern**: CREATE + IMPORT SHIM (not simple shim)
- **Target Location**: Undecided
  - Option 1: `src/investigator/infrastructure/utils/json.py`
  - Option 2: `src/investigator/domain/services/json_utils.py`
- **Complexity**: Low - small utility module

### Recommended Action

**Phase 2a: Create Canonical Modules**
1. Create canonical versions in chosen locations
2. Ensure feature parity
3. Create import shims in utils/
4. Update imports in src/investigator/
5. Verify and archive shims

---

## Category 3: Deferred Complex Migrations

Already identified in Phase 1 summary. These require full module restructuring, not just import updates.

### monitoring.py
- **Size**: 625 lines, 5 classes
- **Imports**: 4 files, 5 imports
- **Target**: `src/investigator/infrastructure/monitoring/`
- **Status**: DEFERRED (requires full module migration)

### event_bus.py
- **Size**: 446 lines
- **Imports**: 2 files, 2 imports
- **Target**: `src/investigator/infrastructure/events/`
- **Status**: DEFERRED (requires full module migration)

---

## Modules with Zero Imports (Low Priority)

These modules have **0 imports** from `src/investigator/` and can remain in utils/ for now:

- `financial_calculators.py` - Helper functions
- `period_utils.py` - Period helpers
- 40+ other utility modules (report generators, visualizers, etc.)

These are stable and not blocking clean architecture adoption.

---

## Migration Path Forward

### Immediate Next Steps (Phase 2)

**Option A: Architectural Decision on Valuation**
1. Decide on valuation architecture direction
2. Either migrate dcf_valuation.py to new architecture OR keep legacy
3. Archive gordon_growth_model.py and utils/valuation/ if confirmed dead

**Option B: Simple Utility Migrations**
1. Create canonical_key_mapper in chosen location
2. Create json_utils in chosen location
3. Apply Phase 1 import shim pattern
4. Archive shims

**Option C: Final Cleanup**
1. Audit entire utils/ directory
2. Identify all unused modules (0 imports)
3. Archive dead code in bulk
4. Document stable modules remaining in utils/

### Recommendation

Start with **Option C** (cleanup) to reduce clutter, then **Option B** (simple utilities), then **Option A** (architectural decisions) as those require more stakeholder input.

---

## Key Findings

1. **Gordon Growth Model Already Migrated**: The `gordon_growth_model.py` in utils/ is dead code. Clean architecture already has it in `src/investigator/domain/services/valuation.py`.

2. **Parallel Valuation Architectures**: Two separate systems exist. Need to decide which is canonical.

3. **utils/valuation/ Directory Unused**: 12 files, 0 imports from src/. Likely candidate for bulk archival.

4. **Import Shim Pattern Works Well**: Phase 1 (7 modules) completed successfully with this approach.

5. **Remaining Migrations Are Different**: No longer simple "canonical version already exists" cases.

---

**Last Updated**: 2025-11-13
**Next Action**: User/architect decision on valuation infrastructure direction
