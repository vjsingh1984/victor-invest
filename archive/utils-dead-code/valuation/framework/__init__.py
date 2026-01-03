"""
Valuation framework orchestrator (legacy location).

NOTE: This package is maintained for MultiModelValuationOrchestrator compatibility.
Most valuation models and helper functions have been migrated to clean architecture:
  - Models: src/investigator/domain/services/valuation/models/
  - Helpers: src/investigator/domain/services/valuation/helpers.py

Phase 6 Migration (2025-11-14):
  - Helper functions migrated to clean architecture
  - Base models archived (now use clean architecture versions)
  - Orchestrator class remains here temporarily (will be migrated in future phase)
"""

# NOTE: MultiModelValuationOrchestrator is imported directly from orchestrator.py
# No need to re-export here since fundamental agent imports directly
