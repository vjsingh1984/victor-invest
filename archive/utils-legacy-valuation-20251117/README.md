# Legacy Valuation Code - Archived 2025-11-17

These files were deprecated after migration to clean architecture.

## Files Archived

### `dcf_valuation.py`
- **Old Location**: `utils/dcf_valuation.py`
- **New Location**: `src/investigator/domain/services/valuation/dcf.py`
- **Reason**: Migrated to clean architecture (domain services)
- **Status**: ❌ NO active imports remaining

### `gordon_growth_model.py`
- **Old Location**: `utils/gordon_growth_model.py`
- **New Location**: Functionality integrated into DCF class
- **Reason**: GGM is now a method within the DCF valuation framework
- **Status**: ❌ NO active imports remaining

## Verification

Both files verified as unused before archival:

```bash
# No imports found in active codebase
grep -r "from utils.dcf_valuation import" src/investigator/domain/agents
# (no results)

grep -r "from utils.gordon_growth_model import" src/investigator/domain/services
# (no results)
```

## Architecture Migration

The codebase has migrated to **clean architecture**:

- **Old (Deprecated)**: `utils/dcf_valuation.py`, `utils/gordon_growth_model.py`
- **New (Active)**: `src/investigator/domain/services/valuation/dcf.py`

All DCF and GGM valuation now uses the NEW implementation. The old files were archived to prevent accidental reintroduction of deprecated code.

## Context

This archival was performed during investigation of SNOW (Snowflake) DCF valuation issues. Root cause analysis revealed:

1. Pre-profitable config module created (`src/investigator/domain/services/pre_profitable_config.py`)
2. Config needed to be integrated into NEW DCF (not old `utils/dcf_valuation.py`)
3. Old DCF files were no longer in use, creating confusion
4. Files archived to enforce clean architecture migration

## Related Documentation

- `/tmp/SNOW_DCF_FIX_ROOT_CAUSE_20251117.md` - Root cause of DCF migration discovery
- `/tmp/SNOW_PRE_PROFITABLE_FIX_IMPLEMENTATION.md` - Implementation plan
- `src/investigator/domain/services/pre_profitable_config.py` - Pre-profitable config module

---

**Archived**: 2025-11-17
**Reason**: Clean architecture migration complete, prevent future confusion
