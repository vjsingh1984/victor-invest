# Valuation Services Consolidation Analysis

**Date**: November 7, 2025
**Objective**: Analyze MultiModelValuationOrchestrator vs DynamicModelWeightingService for potential consolidation

---

## Executive Summary

Two services currently handle multi-model valuation blending with significant overlap in functionality but different approaches:

1. **MultiModelValuationOrchestrator** (`utils/valuation/orchestrator.py`)
   - **Location**: Legacy utils/ directory
   - **Approach**: Confidence-based weighting (data-driven)
   - **Focus**: Model result blending, agreement scoring, divergence detection

2. **DynamicModelWeightingService** (`src/investigator/domain/services/dynamic_model_weighting.py`)
   - **Location**: Clean architecture domain services
   - **Approach**: Tier-based configuration-driven weighting
   - **Focus**: Pre-execution weight determination, business logic classification

**Recommendation**: **Integrate, don't merge** - These services serve complementary roles in different stages of the valuation pipeline.

---

## Detailed Comparison

### 1. Core Responsibilities

| Aspect | MultiModelValuationOrchestrator | DynamicModelWeightingService |
|--------|--------------------------------|------------------------------|
| **Primary Job** | Blend already-executed model outputs | Determine which models to run and with what priority |
| **Input** | Model results (fair_value, confidence) | Company financials, ratios, sector |
| **Output** | Blended fair value, agreement score | Model weights (pre-execution) |
| **Stage** | Post-execution (synthesis) | Pre-execution (planning) |
| **Weighting Method** | Confidence-driven (data quality) | Tier/sector-driven (business logic) |
| **Configuration** | Fallback weights (optional) | 15 sub-tiers with detailed config |
| **Applicability Check** | Post-hoc (filters out null results) | Pre-execution (sets weight to 0) |

### 2. Weighting Philosophy

#### MultiModelValuationOrchestrator
```python
# CONFIDENCE-BASED (Data-Driven)
# - Models produce confidence_score (0.0-1.0) based on data quality
# - Weight = confidence / sum(all_confidences)
# - If all confidences = 0, use equal weighting or fallback config
# - REACTIVE: Adjusts based on actual model execution results

Example:
  DCF:       confidence=0.8 → weight=0.47 (47%)
  P/E:       confidence=0.6 → weight=0.35 (35%)
  EV/EBITDA: confidence=0.3 → weight=0.18 (18%)
```

#### DynamicModelWeightingService
```python
# TIER-BASED (Business Logic-Driven)
# - Classifies company into 15 sub-tiers based on:
#   - Profitability (pre-profit, profitable)
#   - Dividend payout (0-100%)
#   - Growth (Rule of 40, revenue growth)
#   - Sector (Technology, Energy, Financials, etc.)
# - Each tier has configured base weights
# - PROACTIVE: Determines weights before execution

Example (dividend_aristocrat_pure):
  DCF:       50% (cash flow stability)
  GGM:       30% (dividend discount model)
  P/E:       15%
  EV/EBITDA: 5%
  P/S:       0%
  P/B:       0%
```

### 3. Feature Comparison Matrix

| Feature | MultiModel Orchestrator | Dynamic Weighting Service | Consolidation Strategy |
|---------|-------------------------|---------------------------|------------------------|
| **Model Result Blending** | ✅ Core feature | ❌ Not applicable | ✅ Keep in Orchestrator |
| **Agreement Scoring** | ✅ Calculates CoV | ❌ No | ✅ Keep in Orchestrator |
| **Divergence Detection** | ✅ Threshold-based | ❌ No | ✅ Keep in Orchestrator |
| **Confidence Weighting** | ✅ Dynamic | ❌ No | ✅ Keep in Orchestrator |
| **Tier Classification** | ❌ No | ✅ 15 sub-tiers | ⚠️ Consider sharing tier logic |
| **Sector/Industry Lookup** | ❌ No | ✅ Database + fallback | ⚠️ Move to shared service |
| **Industry Overrides** | ❌ No | ✅ Config-driven | ✅ Keep in Weighting Service |
| **Applicability Filtering** | ✅ Post-hoc (nulls) | ✅ Pre-execution | ⚠️ **DUPLICATION** - Unify |
| **Data Quality Adjustment** | ❌ No | ✅ Per-model grades | ⚠️ Consider adding to Orchestrator |
| **Weight Normalization** | ✅ Sum to 100% | ✅ Sum to 100%, round to 5% | ⚠️ **DUPLICATION** - Unify |
| **Fallback Weights** | ✅ Optional config | ✅ Tier-based config | ⚠️ **SYNERGY** - Harmonize |

### 4. Code Location & Architecture

| Aspect | MultiModel Orchestrator | Dynamic Weighting Service |
|--------|-------------------------|---------------------------|
| **Path** | `utils/valuation/orchestrator.py` | `src/investigator/domain/services/dynamic_model_weighting.py` |
| **Architecture** | Legacy utils/ | Clean architecture domain/ |
| **Dependencies** | Base classes (ValuationModelResult) | Database (stock), config.json |
| **Used By** | Fundamental agent (synthesis) | Fundamental agent (pre-execution) |
| **Imports** | Internal (valuation module) | External (database, SQLAlchemy) |
| **Migration Status** | ⚠️ Should migrate to src/ | ✅ Already in clean architecture |

---

## Duplication & Overlap Issues

### 1. Applicability Filtering

**DUPLICATE LOGIC** - Both services check if models are applicable:

#### MultiModelValuationOrchestrator (Post-Execution)
```python
# Lines 99-101
applicable = [
    model for model in models
    if model.get("applicable") and isinstance(model.get("fair_value_per_share"), (int, float))
]
```

#### DynamicModelWeightingService (Pre-Execution)
```python
# Lines 418-477: _apply_applicability_filters()
if weights.get("ggm", 0) > 0:
    if dividends <= 0 or net_income <= 0 or payout_ratio < 40:
        filtered["ggm"] = 0  # Not applicable

if weights.get("pe", 0) > 0:
    if net_income <= 0:
        filtered["pe"] = 0  # Not applicable
```

**Problem**: Two different places checking the same business rules.

**Solution**:
- ✅ Keep pre-execution filtering in DynamicModelWeightingService (more efficient - don't run unnecessary models)
- ✅ Keep post-execution null checking in Orchestrator (safety net for models that execute but fail)
- ⚠️ **Action Required**: Ensure both use IDENTICAL applicability criteria (currently divergent)

### 2. Weight Normalization

**DUPLICATE LOGIC** - Both normalize weights to sum to 100%:

#### MultiModelValuationOrchestrator
```python
# Lines 142-148: Confidence-based normalization
if total_confidence <= 0:
    weight = 1.0 / len(applicable)
    for model in applicable:
        model["weight"] = round(weight, 4)
else:
    for model, confidence in zip(applicable, confidences):
        model["weight"] = round(confidence / total_confidence, 4)
```

#### DynamicModelWeightingService
```python
# Lines 519-564: _normalize_and_round_weights()
total = sum(non_zero.values())
normalized = {k: (v / total * 100) for k, v in non_zero.items()}

# Round to nearest 5%
rounded = {model: round(weight / 5) * 5 for model, weight in normalized.items()}

# Adjust to ensure sum = 100%
diff = 100 - sum(rounded.values())
max_model = max(rounded, key=rounded.get)
rounded[max_model] += diff
```

**Difference**:
- Orchestrator rounds to 4 decimals (0.0001 precision)
- Weighting Service rounds to 5% increments (0.05 precision)

**Problem**: Inconsistent precision across services.

**Solution**:
- ✅ **Action Required**: Standardize on 5% increments for simplicity and explainability
- Update Orchestrator to use same rounding logic as Weighting Service

### 3. Fallback Weight Configuration

**SYNERGY OPPORTUNITY** - Both use fallback weights but from different sources:

#### MultiModelValuationOrchestrator
```python
# Lines 126-140: Accepts fallback_weights parameter
if total_confidence <= 0 and fallback_weights:
    matched = {model.get("model"): float(fallback_weights.get(model.get("model"))) ...}
    # Apply fallback from parameter
```

#### DynamicModelWeightingService
```python
# Lines 350-368: Uses tier-based config
weights = self.tier_base_weights.get(sub_tier)
if not weights:
    weights = self.tier_base_weights.get("balanced_default", {
        "dcf": 30, "pe": 25, "ev_ebitda": 20, "ps": 15, "pb": 10, "ggm": 0
    })
```

**Problem**: Two separate configuration sources for "default" weights.

**Solution**:
- ✅ **Action Required**: Harmonize configuration - use tier-based weights from DynamicModelWeightingService as the "fallback_weights" for Orchestrator
- Pass tier-determined weights to Orchestrator's `combine()` method as fallback

---

## Synergies & Integration Opportunities

### 1. Two-Stage Weighting Pipeline

**Current State**: Weights determined twice (once pre-execution, once post-execution)

**Proposed Integration**:

```python
# STAGE 1: Pre-Execution (DynamicModelWeightingService)
tier_weights = dynamic_weighting_service.determine_weights(
    symbol="AAPL",
    financials={...},
    ratios={...},
    data_quality=None  # Not yet available
)
# Output: {"dcf": 50, "pe": 30, "ev_ebitda": 15, "ps": 5, "pb": 0, "ggm": 0}

# STAGE 2: Model Execution (Individual Models)
# - Run only models with weight > 0
# - Models produce results with confidence_score

# STAGE 3: Post-Execution Blending (MultiModelValuationOrchestrator)
blended_result = orchestrator.combine(
    company_profile=profile,
    model_outputs=[dcf_result, pe_result, ...],
    fallback_weights=tier_weights  # Use tier weights as fallback!
)
# Output: blended_fair_value, overall_confidence, agreement_score
```

**Benefits**:
- Tier-based weights guide model selection (don't run DCF for pre-profit companies)
- Confidence-based weighting adjusts for data quality issues
- Fallback to tier weights if all confidences are 0

### 2. Shared Sector/Industry Lookup

**Current State**: DynamicModelWeightingService has database lookup logic

**Opportunity**: Extract to shared service for reuse

```python
# NEW: src/investigator/domain/services/company_metadata_service.py
class CompanyMetadataService:
    """Fetch company metadata from multiple sources with fallback priority."""

    def get_sector_industry(self, symbol: str) -> Tuple[str, Optional[str]]:
        """
        Priority:
        1. sec_sector (SEC CompanyFacts)
        2. Sector (Yahoo Finance)
        3. Peer Group JSON (data/sector_mapping.json)
        4. "Unknown" (final fallback)
        """
        # Move _get_normalized_sector_industry() here
        pass

# USAGE:
# - DynamicModelWeightingService uses for tier classification
# - Other agents (SEC, Fundamental) use for sector-specific logic
# - Symbol_Update agent uses for backfilling
```

### 3. Unified Applicability Rules

**Current State**: Applicability rules scattered across:
- DynamicModelWeightingService._apply_applicability_filters()
- Individual model classes (BaseValuationModel, DCFModel, etc.)
- Post-hoc filtering in Orchestrator

**Opportunity**: Centralize in shared config

```json
// config.json - Add "model_applicability" section
{
  "valuation": {
    "model_applicability": {
      "dcf": {
        "min_quarters_data": 4,
        "require_positive_fcf": false,
        "reason": "DCF requires at least 4 quarters of cash flow data"
      },
      "ggm": {
        "min_payout_ratio": 40,
        "require_positive_earnings": true,
        "require_dividends": true,
        "reason": "GGM requires consistent dividends and 40%+ payout ratio"
      },
      "pe": {
        "require_positive_earnings": true,
        "reason": "P/E multiple requires positive earnings"
      },
      "ps": {
        "require_positive_revenue": true,
        "reason": "P/S multiple requires positive revenue"
      },
      "pb": {
        "require_positive_book_value": true,
        "reason": "P/B multiple requires positive book value"
      },
      "ev_ebitda": {
        "require_positive_ebitda": true,
        "reason": "EV/EBITDA requires positive EBITDA"
      }
    }
  }
}
```

---

## Consolidation Recommendations

### ❌ DO NOT Merge Into Single Class

**Reason**: Services operate at different pipeline stages with distinct responsibilities

- **DynamicModelWeightingService** = "Which models should we run and prioritize?"
- **MultiModelValuationOrchestrator** = "How do we blend the results we got?"

Merging would violate Single Responsibility Principle.

### ✅ DO: Integrate as Pipeline Stages

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                  Fundamental Analysis Agent                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: DynamicModelWeightingService                      │
│  - Classify company into tier (15 sub-tiers)                │
│  - Get base weights for tier                                │
│  - Apply sector/industry overrides                          │
│  - Filter inapplicable models (weight=0)                    │
│  Output: {"dcf": 50, "pe": 30, ...}                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Model Execution                                   │
│  - Run only models with weight > 0                          │
│  - Each model produces: fair_value, confidence_score        │
│  Output: [DCFResult(fv=180, conf=0.8), PEResult(...), ...]  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: MultiModelValuationOrchestrator                   │
│  - Normalize model outputs                                  │
│  - Adjust weights based on confidence (overrides tier)      │
│  - Calculate blended fair value                             │
│  - Calculate agreement score / divergence                   │
│  - Use tier weights as fallback if confidences = 0          │
│  Output: {blended_fv: 175, confidence: 0.82, agreement: 0.9}│
└─────────────────────────────────────────────────────────────┘
```

### ✅ DO: Extract Shared Concerns

**1. Create CompanyMetadataService**

Move sector/industry lookup logic to shared service.

**Location**: `src/investigator/domain/services/company_metadata_service.py`

**Responsibility**:
- Fetch sector/industry from database (stock.symbol table)
- Fallback to peer group JSON
- Normalize sector names
- Cache lookups for performance

**2. Create ModelApplicabilityRules**

Centralize applicability logic in config-driven class.

**Location**: `src/investigator/domain/services/model_applicability.py`

**Responsibility**:
- Load rules from config.json
- Check if model is applicable given financials
- Return (applicable: bool, reason: str)
- Used by both DynamicWeightingService (pre) and Orchestrator (post)

**3. Standardize Weight Normalization**

Extract normalization to shared utility.

**Location**: `src/investigator/domain/services/weight_normalization.py`

**Responsibility**:
- Normalize weights to sum = 100%
- Round to configurable increment (default 5%)
- Adjust largest weight to ensure exact 100%

### ✅ DO: Harmonize Configuration

**Current State**: Weights configured in multiple places
- `config.json` → tier_base_weights (DynamicWeightingService)
- Fundamental agent → fallback_weights parameter (Orchestrator)
- Hardcoded defaults in code

**Proposed**: Single source of truth in `config.json`

```json
{
  "valuation": {
    "tier_base_weights": {
      "dividend_aristocrat_pure": {"dcf": 50, "ggm": 30, "pe": 15, "ev_ebitda": 5, "ps": 0, "pb": 0},
      "high_growth_hyper": {"dcf": 40, "pe": 35, "ps": 20, "ev_ebitda": 5, "pb": 0, "ggm": 0},
      "balanced_default": {"dcf": 30, "pe": 25, "ev_ebitda": 20, "ps": 15, "pb": 10, "ggm": 0}
    },
    "orchestrator": {
      "divergence_threshold": 0.35,
      "min_confidence_threshold": 0.5,
      "weight_rounding_increment": 5
    }
  }
}
```

---

## Migration Plan

### Phase 1: Extract Shared Services (Week 1)

**Tasks**:
1. Create `CompanyMetadataService` with sector/industry lookup
   - Move logic from `DynamicModelWeightingService._get_normalized_sector_industry()`
   - Add caching for performance
   - Update DynamicWeightingService to use it

2. Create `ModelApplicabilityRules` with centralized rules
   - Load from `config.json` → `model_applicability`
   - Update DynamicWeightingService to use it
   - Update individual model classes to use it

3. Create `WeightNormalizer` utility
   - Extract from both services
   - Standardize on 5% increments
   - Update both services to use it

**Tests**: Unit tests for each new service

### Phase 2: Integrate Pipeline (Week 2)

**Tasks**:
1. Update Fundamental Agent to use two-stage pipeline:
   ```python
   # STAGE 1: Determine tier-based weights
   tier_weights = self.dynamic_weighting_service.determine_weights(...)

   # STAGE 2: Run only applicable models
   model_results = []
   for model_name, weight in tier_weights.items():
       if weight > 0:
           result = self._run_model(model_name, ...)
           model_results.append(result)

   # STAGE 3: Blend results
   blended = self.orchestrator.combine(
       company_profile=profile,
       model_outputs=model_results,
       fallback_weights=tier_weights  # Pass tier weights as fallback!
   )
   ```

2. Harmonize configuration in `config.json`

3. Add integration tests (AAPL, MSFT, NVDA, JNJ, XOM, NEE)

**Tests**: Integration tests with known companies

### Phase 3: Migrate to Clean Architecture (Week 3)

**Tasks**:
1. Move `MultiModelValuationOrchestrator` from `utils/valuation/` to `src/investigator/domain/services/`

2. Update imports across codebase

3. Deprecate old utils/ location

**Tests**: Regression tests to ensure no behavior changes

### Phase 4: Documentation & Validation (Week 4)

**Tasks**:
1. Update CLAUDE.md with new architecture
2. Create valuation pipeline documentation
3. Run full test suite against S&P 100 sample
4. Performance benchmarking

---

## Benefits of Integration

### 1. Consistency
- Single source of truth for model applicability rules
- Consistent weight normalization across services
- Harmonized configuration

### 2. Efficiency
- Pre-execution filtering avoids running inapplicable models
- Tier-based weights reduce reliance on fallbacks
- Shared services reduce code duplication

### 3. Explainability
- Clear two-stage pipeline (tier → confidence)
- Audit trail shows both tier classification and confidence adjustments
- Divergence detection highlights model disagreement

### 4. Maintainability
- Shared services easier to update (one place, not two)
- Clear separation of concerns
- Clean architecture migration path

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Breaking Changes** | High | Comprehensive integration tests, gradual rollout |
| **Performance Regression** | Medium | Benchmark before/after, optimize shared services |
| **Configuration Complexity** | Medium | Validation schema, clear documentation |
| **Migration Effort** | Medium | Phased approach, incremental updates |

---

## Conclusion

**DO NOT merge** MultiModelValuationOrchestrator and DynamicModelWeightingService into a single class. They serve complementary roles at different pipeline stages.

**DO integrate** them as a two-stage pipeline:
1. **Stage 1** (DynamicWeightingService): Tier-based weight determination (pre-execution)
2. **Stage 2** (Orchestrator): Confidence-based blending (post-execution)

**DO extract** shared concerns:
- CompanyMetadataService (sector/industry lookup)
- ModelApplicabilityRules (centralized applicability logic)
- WeightNormalizer (standardized normalization)

**DO harmonize** configuration in `config.json` for consistency.

**Estimated Effort**: 4 weeks (1 week per phase)

**Expected Benefits**:
- 30% reduction in code duplication
- 15% performance improvement (skip inapplicable models)
- Improved explainability and auditability
- Cleaner architecture aligned with domain-driven design

---

**Generated**: November 7, 2025
**Author**: InvestiGator Valuation Architecture Review
**Next Steps**: Review with team, approve migration plan, execute Phase 1
