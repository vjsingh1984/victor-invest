# Codebase Organization Recommendations for InvestiGator

This document summarizes the recommendations for improving the organization of the InvestiGator codebase, based on an analysis of dead code, unreferenced code, and overall directory structure.

## 1. Dead and Unreferenced Code

**Summary from Vulture Scan:**
The `vulture` scan identified a significant amount of unused imports, functions, classes, and variables across various modules, particularly in `admin/dashboard.py`, `api/main.py`, `cli_orchestrator.py`, `config.py`, `dao/sec_bulk_dao.py`, `data/models.py`, `legacy_tests/`, `models/database.py`, `patterns/`, `scripts/`, `src/investigator/`, `utils/`, and `yahoo_technical.py`.

**Recommendation:**
*   **Systematic Review and Removal:** Conduct a thorough, module-by-module review of all items flagged by `vulture`. For each flagged item:
    *   Verify if it is indeed unused.
    *   If unused, safely remove it.
    *   If it's a false positive (e.g., used dynamically or via introspection), add a `# vulture ignore` comment to suppress future warnings.
*   **Prioritize Core Modules:** Start with core application logic (`src/investigator/application`, `src/investigator/domain`) and API endpoints (`api/main.py`) to ensure critical paths are clean.
*   **Refactor `config.py`:** The `config.py` file shows many unused variables. Review and consolidate configuration settings, removing any that are no longer active or relevant.
*   **Address `legacy_tests`:** The `legacy_tests` directory contains a large number of unused elements. This reinforces the idea that these tests are likely obsolete and should be either updated, integrated into the main `tests` suite, or removed entirely.

## 2. Top-Level Directory Structure Improvements

**Current State:**
The top-level directory is cluttered with a mix of configuration files, scripts, data files, documentation, and application code.

**Implemented Changes (as of 2025-11-09):**
The following structural changes have been applied:
*   **Configuration:** Consolidated `config.json`, `config.py`, `config.yaml`, `.env.example`, and `.env.template` into a new `config/` directory.
*   **Scripts:** Moved various shell and Python scripts from the root into a new `scripts/` directory.
*   **Data:** Moved various data files and ticker-specific directories (e.g., `AAPL/`, `AMZN/`) into a new `data/` directory.
*   **Documentation:** Moved various Markdown and AsciiDoc documentation files into a new `docs/` directory.
*   **CLI Entry Point:** Moved `cli_orchestrator.py` to `src/investigator/cli/orchestrator.py`.
*   **Patterns Consolidation:** Moved contents of `patterns/` into `src/investigator/`.
*   **Obsolete Directories Removed:** `archive/` and `legacy_tests/` have been removed.

**Further Recommendations for Directory Structure:**

*   **`src/investigator/` Refinement:**
    *   Review the contents moved from `patterns/` into `src/investigator/`. Ensure they are logically placed within `application/`, `domain/`, `infrastructure/`, or `interfaces/`. Create new subdirectories as needed (e.g., `strategies/`, `adapters/`, `services/`) to maintain clear separation of concerns.
    *   Ensure `__init__.py` files are correctly placed in all new and existing Python package directories within `src/investigator/`.
*   **`utils/` Directory Review:** The `utils/` directory often becomes a dumping ground. Review its contents and move modules into more specific locations within `src/investigator/` if they belong to a particular domain or infrastructure layer. Only truly generic, project-wide utilities should remain in `utils/`.
*   **`models/` Directory:** The `models/` directory at the root level (containing `database.py`) should ideally be moved into `src/investigator/domain/models/` to align with a clean architecture pattern where domain models reside within the domain layer.
*   **`dao/` Directory:** Similar to `models/`, the `dao/` directory should be moved to `src/investigator/infrastructure/persistence/` or `src/investigator/infrastructure/dao/` to reflect its role in data access.
*   **`tests/` Directory:**
    *   Integrate any remaining relevant tests from `legacy_tests/` into the main `tests/` directory, following the existing test structure (e.g., `tests/unit/`, `tests/integration/`).
    *   Ensure tests mirror the new `src/investigator/` structure for easier navigation and maintenance.
*   **Root Level Cleanup:** After the above moves, the root directory should ideally only contain project-level configuration files (e.g., `pyproject.toml`, `requirements.txt`, `Makefile`, `Dockerfile`, `.gitignore`), top-level documentation (`README.md`), and possibly build/deployment scripts.

## Next Steps:

1.  **Review and Refine:** Carefully review the `vulture` output and the new directory structure.
2.  **Implement Dead Code Removal:** Systematically remove identified dead code.
3.  **Further Structural Refinements:** Apply the "Further Recommendations for Directory Structure" outlined above.
4.  **Update Imports:** After moving files, update all affected import statements across the codebase.
5.  **Run Tests:** Thoroughly run all tests to ensure no functionality was broken during refactoring.
6.  **Update Documentation:** Reflect the new structure in any relevant documentation (e.g., `README.md`, developer guides).
