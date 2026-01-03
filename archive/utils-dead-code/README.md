# Dead Code Archive - utils/ Modules

**Archived Date**: 2025-11-13
**Reason**: Zero imports - completely unused code
**Verification**: Comprehensive audit found 0 imports from any active code (src/investigator/, patterns/, tests/)

---

## Archived Modules

### data_collector.py (32.5K)
- **Imports**: 0 total
- **Purpose**: Data collection utility
- **Reason**: Never integrated into active codebase

### fred_data_fetcher.py (19.7K)
- **Imports**: 0 total
- **Purpose**: FRED economic data fetcher
- **Reason**: Never integrated into active codebase

### gordon_growth_model.py (21.8K) ⭐ **DUPLICATE**
- **Imports**: 0 total
- **Purpose**: Gordon Growth Model valuation
- **Reason**: **ALREADY MIGRATED** to `src/investigator/domain/services/valuation.py`
- **Notes**: This is the canonical GordonGrowthModel implementation used by clean architecture. The utils/ version is dead code.

### peer_group_analyzer.py (9.9K)
- **Imports**: 0 total
- **Purpose**: Peer group analysis utility
- **Reason**: Never integrated into active codebase

### prompt_manager_enhanced.py (16.5K)
- **Imports**: 0 total
- **Purpose**: Enhanced prompt management
- **Reason**: Never integrated into active codebase

---

## Verification Method

**Audit Command**:
```bash
python3 /tmp/audit_utils.py
python3 /tmp/check_total_usage.py
```

**Results**: All 5 modules showed:
- **Imports from src/investigator/**: 0
- **Total imports** (including tests, patterns, utils): 0

**Audit Date**: 2025-11-13

---

## Rollback Procedure

If any module needs to be restored:

```bash
git mv archive/utils-dead-code/MODULE_NAME.py utils/MODULE_NAME.py
```

Git history is preserved via `git mv`, so rollback is safe and easy.

---

## Related Documentation

- **Migration Summary**: `docs/UTILS_MIGRATION_SUMMARY.md`
- **Phase 2 Investigation**: `docs/UTILS_MIGRATION_PHASE2_INVESTIGATION.md`
- **Phase 1 Archive**: `archive/legacy-utils/`

---

**Last Updated**: 2025-11-13
**Phase**: Cleanup (Phase 2 - Options A & C)

---

## Valuation Modules (Phase 2-C)

**Archived Date**: 2025-11-13 (Phase 2-C)
**Total Size**: ~146 KB
**Verification**: Comprehensive audit found 0 imports across entire codebase

### Top-Level Valuation Modules

#### valuation/ (insurance_valuation.py, sector_valuation_router.py, valuation_adjustments.py, valuation_table_formatter.py)
- **insurance_valuation.py** (10.4 KB) - Insurance sector-specific valuation logic
- **sector_valuation_router.py** (10.5 KB) - Routes to sector-specific valuation models
- **valuation_adjustments.py** (15.2 KB) - Quality/risk adjustments to valuations
- **valuation_table_formatter.py** (17.9 KB) - Format valuation results as tables
- **Imports**: 0 total for all 4 modules
- **Purpose**: Sector-specific and presentation utilities for valuation
- **Reason**: Never integrated into clean architecture; unused framework

### Valuation Framework (framework/)

**Directory**: `archive/utils-dead-code/valuation/framework/` (entire `utils/valuation/` directory)
**Size**: ~92 KB

#### Framework Core:
- `base_valuation_model.py` (3.4 KB) - Abstract base class for valuation models
- `company_profile.py` (4.2 KB) - Company profile data structure
- `orchestrator.py` (12 KB) - Valuation model orchestration pattern
- `__init__.py` (0.7 KB) - Framework exports

#### Model Implementations (framework/models/):
- `common.py` (2.3 KB) - Common utilities for models
- `dcf_model.py` (5.8 KB) - DCF model (NOT the active dcf_valuation.py)
- `ev_ebitda_model.py` (8.1 KB) - EV/EBITDA multiple valuation
- `ggm_model.py` (5.6 KB) - Gordon Growth Model implementation
- `pb_multiple_model.py` (5.7 KB) - Price/Book multiple valuation
- `pe_multiple_model.py` (9.5 KB) - P/E multiple valuation
- `ps_multiple_model.py` (6.0 KB) - Price/Sales multiple valuation
- `__init__.py` (0.7 KB) - Model exports

**Notes**:
- This framework represents an earlier architectural approach using base classes and model orchestration
- The ACTIVE DCF implementation is `utils/dcf_valuation.py` (109 KB), NOT `framework/models/dcf_model.py` (5.8 KB)
- Framework was never integrated with clean architecture agents
- All valuation logic now in domain services or being migrated there

**Investigation Document**: `docs/PHASE2C_VALUATION_INVESTIGATION.md`

---

---

## Phase 3-A Modules (2025-11-13)

**Archived Date**: 2025-11-13 (Phase 3-A)
**Total Size**: ~155.5 KB
**Verification**: Comprehensive audit found 0 imports across entire codebase

### Report Generators (report_generators/)

#### comprehensive_reports.py (27.0 KB)
- **Imports**: 0 total
- **Purpose**: Comprehensive report generation
- **Reason**: Never integrated into active codebase

#### peer_group_reports.py (10.4 KB)
- **Imports**: 0 total
- **Purpose**: Peer group comparison reports
- **Reason**: Never integrated into active codebase

### Vector Database (vector_db/)

#### event_analyzer.py (25.0 KB)
- **Imports**: 0 total
- **Purpose**: Vector-based event analysis
- **Reason**: Vector DB feature never implemented

#### vector_engine.py (19.3 KB)
- **Imports**: 0 total
- **Purpose**: Vector database engine
- **Reason**: Vector DB feature never implemented

#### vector_cache_handler.py (17.9 KB)
- **Imports**: 0 total
- **Purpose**: Cache handler for vector data
- **Reason**: Vector DB feature never implemented

### Monitoring (monitoring/)

#### cache_monitor.py (18.0 KB)
- **Imports**: 0 total
- **Purpose**: Cache monitoring utilities
- **Reason**: Superseded by monitoring.py (active module)

#### cache_performance_monitor.py (8.1 KB)
- **Imports**: 0 total
- **Purpose**: Cache performance metrics
- **Reason**: Superseded by monitoring.py (active module)

#### cache_usage_monitor.py (7.3 KB)
- **Imports**: 0 total
- **Purpose**: Cache usage tracking
- **Reason**: Superseded by monitoring.py (active module)

### Analysis Runners (analysis_runners/)

#### enhanced_analysis.py (20.5 KB)
- **Imports**: 0 total
- **Purpose**: Enhanced analysis orchestration
- **Reason**: Never integrated into active codebase

### Data Processing (data_processing/)

#### peer_group_classifier.py (1.8 KB)
- **Imports**: 0 total
- **Purpose**: Peer group classification
- **Reason**: Never integrated into active codebase

**Investigation Document**: `docs/PHASE3_INVESTIGATION.md` (TBD)

---

**Last Updated**: 2025-11-13
**Phases**: Cleanup (Phase 2-A), Valuation Cleanup (Phase 2-C), Dead Code Cleanup (Phase 3-A)
