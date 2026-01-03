# SymbolUpdate Agent - Valuation Data Storage Analysis

**Date**: November 7, 2025
**Purpose**: Analyze current valuation data storage and identify improvements needed

---

## Executive Summary

The SymbolUpdateAgent is responsible for persisting fundamental metrics and valuation results to the `symbol` table in the `stock` database. Currently it stores:

- Individual model fair values (DCF, GGM, PE, PS, PB, EV/EBITDA)
- Blended fair value
- Model quality metrics (confidence, agreement score)
- The complete multi_model_summary in a JSONB column

**Current Issue**: The `valuation_models_json` JSONB column is being populated, but **model weights** (the key output of the two-stage valuation pipeline) are not being explicitly captured in a queryable format.

---

## Current Implementation Analysis

### Symbol Table Structure

The `symbol` table has these valuation-related columns:

**Individual Fair Values** (numeric):
```sql
fair_value_blended      numeric(10,2)   -- Primary blended fair value
fair_value_dcf          numeric(10,2)   -- DCF model
fair_value_ggm          numeric(10,2)   -- Gordon Growth Model
fair_value_ps           numeric(10,2)   -- Price/Sales multiple
fair_value_pe           numeric(10,2)   -- Price/Earnings multiple
fair_value_pb           numeric(10,2)   -- Price/Book multiple
fair_value_ev_ebitda    numeric(10,2)   -- EV/EBITDA multiple
```

**Model Quality Metrics** (numeric):
```sql
model_agreement_score   numeric(5,4)    -- How much models agree (0-1)
model_confidence        numeric(5,4)    -- Overall confidence (0-1)
applicable_models       integer         -- Count of applicable models
divergence_flag         boolean         -- High divergence warning
```

**DCF-Specific Metrics**:
```sql
wacc                    numeric(6,4)    -- Weighted Average Cost of Capital
terminal_growth_rate    numeric(5,4)    -- Terminal growth assumption
dcf_projection_years    integer         -- Projection period
```

**Company Classification**:
```sql
rule_of_40_score        numeric(6,2)    -- Growth + Profitability score
rule_of_40_classification varchar(20)   -- Classification tier
```

**Comprehensive Storage** (JSONB):
```sql
valuation_models_json   jsonb           -- Full multi-model summary
```

---

## What's Currently Stored in JSONB

From `symbol_update.py` line 347:
```python
# Store full JSONB for detailed analysis
update_data["valuation_models_json"] = multi_model_summary
```

The `multi_model_summary` dict contains:
- `blended_fair_value`: Final weighted average fair value
- `overall_confidence`: Overall confidence score
- `model_agreement_score`: How much models agree
- `applicable_models`: Count of applicable models
- `divergence_flag`: Boolean for high divergence
- `models`: Array of individual model results
  - Each model has: `model`, `fair_value_per_share`, `applicable`, `confidence`, `weight`
- `notes`: Array of warning/info messages
- `fallback_applied`: Boolean indicating tier-based fallback used

**PROBLEM**: The current code at line 347 stores `multi_model_summary`, but this dict is extracted from `fundamental.get("multi_model_summary", {})` which may be empty or incomplete depending on the agent's output format.

---

## What's Missing

### 1. Model Weights Not Explicitly Stored

The most critical missing data is **individual model weights**. Currently:

```python
# From line 308-344 in symbol_update.py
models = multi_model_summary.get("models", [])
for model in models:
    # ... extracts fair_value_per_share
    # ❌ Does NOT extract model.get("weight")
```

**Why This Matters**:
- Model weights are the PRIMARY OUTPUT of the two-stage valuation pipeline
- They show which models were most influential (tier-based + confidence-adjusted)
- Critical for understanding why a particular blended FV was calculated
- Needed for auditing and explaining valuation results

### 2. Tier Classification Not Stored

The company tier (from DynamicModelWeightingService) is not captured:
- `tier_name` (e.g., "pre_profit_negative_ebitda", "dividend_aristocrat_pure", "high_growth_hyper")
- This is logged but not persisted to the database

### 3. Fallback Indicators Not Stored

Whether tier-based fallback weights were used is in JSONB but not as a dedicated column for easy querying.

### 4. Dynamic Weighting Context Missing

The tier-based weights (pre-execution stage 1) vs confidence-adjusted weights (post-execution stage 2) are not distinguished.

---

## Recommended Enhancements

### Option 1: Add Dedicated Weight Columns (Simple)

Add columns to `symbol` table:
```sql
ALTER TABLE symbol ADD COLUMN weight_dcf numeric(5,2);
ALTER TABLE symbol ADD COLUMN weight_pe numeric(5,2);
ALTER TABLE symbol ADD COLUMN weight_ps numeric(5,2);
ALTER TABLE symbol ADD COLUMN weight_pb numeric(5,2);
ALTER TABLE symbol ADD COLUMN weight_ggm numeric(5,2);
ALTER TABLE symbol ADD COLUMN weight_ev_ebitda numeric(5,2);
ALTER TABLE symbol ADD COLUMN tier_classification varchar(50);
ALTER TABLE symbol ADD COLUMN fallback_weights_used boolean DEFAULT false;
```

**Pros**:
- Easy to query: `SELECT ticker, weight_dcf, weight_pe FROM symbol WHERE weight_dcf > 30`
- Simple analytics: Average weights by sector, tier, etc.
- Fast filtering

**Cons**:
- Schema changes required
- More columns (model_name × 6 = 6 new columns)
- Rigid structure (adding new models requires schema change)

### Option 2: Enhanced JSONB Storage (Flexible)

Keep JSONB as primary storage but ensure it contains **complete** data:

```json
{
  "blended_fair_value": 833.96,
  "overall_confidence": 0.67,
  "model_agreement_score": 0.927,
  "applicable_models": 5,
  "divergence_flag": false,
  "fallback_applied": false,
  "tier_classification": "pre_profit_negative_ebitda",
  "tier_weights": {
    "dcf": 30,
    "pe": 25,
    "ps": 15,
    "pb": 10,
    "ev_ebitda": 20
  },
  "models": [
    {
      "model": "dcf",
      "fair_value_per_share": 833.96,
      "applicable": true,
      "confidence": 0.75,
      "weight": 0.30,  // ✅ CRITICAL: Final weight after confidence adjustment
      "tier_weight": 0.30,  // ✅ NEW: Original tier-based weight
      "assumptions": {
        "wacc": 0.0945,
        "terminal_growth": 0.025,
        "projection_years": 10
      }
    },
    // ... other models
  ],
  "notes": ["Some warning message"]
}
```

**Pros**:
- Flexible schema (easy to add new models/fields)
- Complete audit trail (tier weights vs final weights)
- PostgreSQL JSONB supports indexing and querying: `WHERE valuation_models_json->>'tier_classification' = 'high_growth'`
- No schema migrations needed

**Cons**:
- Slightly slower queries than dedicated columns
- Requires JSONB query syntax knowledge
- Need to ensure data completeness at extraction time

### Option 3: Hybrid Approach (Recommended)

**Dedicated columns for critical metrics** (fast queries):
```sql
tier_classification         varchar(50)
fallback_weights_used       boolean
```

**JSONB for complete detail**:
- Full model array with weights, tier_weights, assumptions
- Notes, metadata, timestamps

**Code Change in `_extract_metrics()`**:
```python
# Line 347 (existing)
update_data["valuation_models_json"] = multi_model_summary

# NEW: Extract tier classification
tier_classification = multi_model_summary.get("tier_classification")
if tier_classification:
    update_data["tier_classification"] = str(tier_classification)

# NEW: Extract fallback flag
fallback_applied = multi_model_summary.get("fallback_applied")
if fallback_applied is not None:
    update_data["fallback_weights_used"] = bool(fallback_applied)

# ENHANCED: Extract model weights (lines 308-344)
models = multi_model_summary.get("models", [])
for model in models:
    if not isinstance(model, dict):
        continue
    model_name = model.get("model", "").lower()

    # Existing: fair value extraction
    fair_value = model.get("fair_value_per_share")
    if fair_value and fair_value > 0 and model.get("applicable"):
        # ... existing code
        pass

    # NEW: Extract and store weights in JSONB
    # (Already in JSONB if multi_model_summary is complete)
```

---

## Verification Checklist

To verify the SymbolUpdateAgent is storing complete valuation data:

1. **Check multi_model_summary Source**:
   - ✅ Is `fundamental_analysis["multi_model_summary"]` populated?
   - ✅ Does it contain the `models` array with `weight` field?
   - ✅ Does it contain `tier_classification`?

2. **Check _extract_metrics Logic**:
   - ✅ Line 347: Is `multi_model_summary` extracted correctly?
   - ✅ Lines 308-344: Are model weights extracted from `model.get("weight")`?
   - ❌ **MISSING**: Tier classification extraction
   - ❌ **MISSING**: Fallback flag extraction

3. **Check Database After Update**:
   ```sql
   SELECT
     ticker,
     valuation_models_json->>'tier_classification' as tier,
     valuation_models_json->>'fallback_applied' as fallback_used,
     valuation_models_json->'models'->0->>'model' as first_model,
     valuation_models_json->'models'->0->>'weight' as first_weight
   FROM symbol
   WHERE ticker = 'MSFT';
   ```

---

## Implementation Plan

### Phase 1: Verify Current Data Flow (CURRENT TASK)

1. ✅ Analyze `symbol_update.py` code
2. ✅ Check symbol table schema
3. ⏳ Trace data flow from FundamentalAgent → SymbolUpdateAgent
4. ⏳ Verify `multi_model_summary` structure in actual analysis output

### Phase 2: Add Missing Extractions

1. Update `_extract_metrics()` to capture:
   - Tier classification (if available in multi_model_summary)
   - Fallback flag (if available)
   - Individual model weights (ensure they're in JSONB)

2. Optionally add dedicated columns:
   ```sql
   ALTER TABLE symbol ADD COLUMN tier_classification varchar(50);
   ALTER TABLE symbol ADD COLUMN fallback_weights_used boolean DEFAULT false;
   ```

### Phase 3: Test with Real Analysis

1. Run analysis: `python3 cli_orchestrator.py analyze AAPL -m standard --force-refresh`
2. Check database:
   ```sql
   SELECT ticker, valuation_models_json FROM symbol WHERE ticker = 'AAPL';
   ```
3. Verify JSONB contains:
   - tier_classification
   - fallback_applied
   - models array with weight field

---

## Current Status

**Symbol Table Columns**: ✅ Adequate (has JSONB + key numeric columns)
**SymbolUpdateAgent Code**: ⚠️ Partially Complete
- ✅ Stores multi_model_summary in JSONB (line 347)
- ✅ Extracts individual fair values (lines 308-344)
- ✅ Extracts model quality metrics (lines 291-305)
- ❌ Does NOT extract tier_classification
- ❌ Does NOT extract fallback_applied flag
- ❓ Unknown if model weights are in multi_model_summary (need to verify)

**Next Action**: Verify the structure of `multi_model_summary` in actual analysis output to confirm what data is available for extraction.

---

**Analysis By**: InvestiGator Valuation Pipeline Team
**Status**: Investigation Phase - Verification Needed
