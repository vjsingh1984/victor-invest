# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CONTRIBUTING.md with comprehensive contribution guidelines
- Documentation index at docs/INDEX.md for easier navigation
- Consolidated fiscal year handling guide (FISCAL_YEAR_HANDLING.md)
- Clean architecture migration status document (CLEAN_ARCHITECTURE_MIGRATION.md)

### Changed
- Consolidated documentation from 76 files to 14 files (~82% reduction)
- Removed obsolete analysis documents from analysis/ folder
- Removed duplicate config.py (kept src/investigator/config/config.py)
- Removed duplicate .env.template (kept config/.env.example)
- Updated parquet_cache_handler.py to use new config import path

### Removed
- 62 obsolete documentation files from docs/:
  - Session summaries (dated 2025-11-12, 2025-11-13)
  - Phase-specific migration tracking docs
  - Fiscal year analysis files (consolidated)
  - Architecture duplication analysis files
  - Obsolete integration plans and test result documents
  - Temporary investigation files
- 32 obsolete analysis documents from analysis/:
  - Historical bug analysis documents
  - Implementation summaries
  - Dated analysis files
- Old config/config.py (95KB duplicate)
- config/.env.template (duplicate of .env.example)
- config/macos/ folder (user-specific files)

### Fixed
- Updated import in parquet_cache_handler.py from old config path to new clean architecture path

## [Previous Releases]

### Phase 5: SRP Extraction
- Extracted TrendAnalyzer, DataQualityAssessor, DeterministicAnalyzer
- 183 new tests added
- All code following Single Responsibility Principle

### Phase 4: Configuration & Critical Fixes
- Configuration validation with 34 tests
- Fiscal year handling fixes
- Data quality improvements

### Phase 3: Database Integration
- Fiscal period-aware caching
- Database persistence layer
- TTL enforcement

### Phase 2: Utils Migration
- Migrated 17 modules from utils/ to clean architecture
- Removed 255KB of dead code
- Created import shims for backward compatibility

### Phase 1: Foundation
- Data normalization (snake_case, rounding)
- Fiscal period detection service
- Cache key standardization

---

## Documentation Structure

### Core Documentation
- README.adoc - Main project overview
- ARCHITECTURE.md - Comprehensive architecture documentation
- DEVELOPER_GUIDE.adoc - Development setup and guidelines
- INDEX.md - Documentation index (NEW)

### Migration & Architecture
- CLEAN_ARCHITECTURE_MIGRATION.md - Clean architecture status (NEW)
- MIGRATION_GUIDE.md - How to migrate code

### Specialized Guides
- FISCAL_YEAR_HANDLING.md - Fiscal year handling guide (NEW)
- VALUATION_ASSUMPTIONS.md - Valuation methodology
- VALUATION_CONFIGURATION.md - Valuation configuration
- CLI_DATA_COMMANDS.md - CLI command reference
- OPERATIONS_RUNBOOK.md - Operational procedures

---

**For detailed documentation, see [docs/INDEX.md](docs/INDEX.md)**
