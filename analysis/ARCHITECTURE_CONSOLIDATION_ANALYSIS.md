# SEC Data Architecture Analysis & Consolidation Recommendation

**Date**: 2025-11-04
**Issue**: DCF returning $0.0M FCF due to empty `quarterly_metrics` table
**Root Cause**: Architectural duplication between two quarterly data tables

---

## Current Architecture - Dual Table Problem

### Table 1: `sec_companyfacts_processed` ✅ **HAS DATA**
**Purpose**: Official processed quarterly/annual data from `utils/sec_data_processor.py`

**Schema**: 42 columns with explicit fields
- Financial metrics as COLUMNS: `operating_cash_flow`, `capital_expenditures`, `free_cash_flow`, etc.
- All ratios pre-calculated: `current_ratio`, `debt_to_equity`, `roa`, `roe`, etc.
- Quality metadata: `data_quality_score`, `extraction_version`
- Lineage tracking: `raw_data_id` → `sec_companyfacts_raw`

**Data Status for AAPL**:
```
66 quarters populated
2025-Q3: OCF=$91.443B, CapEx=$6.539B, FCF=$84.904B ✅
2025-Q2: OCF=$62.585B, CapEx=$4.388B, FCF=$58.197B ✅
2025-Q1: OCF=$39.895B, CapEx=$2.392B, FCF=$37.503B ✅
```

**Primary Key**: `(symbol, fiscal_year, fiscal_period, adsh)`
**Populated By**: `SECDataProcessor.save_processed_data()` (utils/sec_data_processor.py:460)
**Last Updated**: 2025-11-04

---

### Table 2: `quarterly_metrics` ❌ **EMPTY DATA**
**Purpose**: Legacy flexible quarterly storage with JSONB `metrics_data`

**Schema**: 9 columns with JSONB blob
- `metrics_data` JSONB: Supposed to contain all financial data
- Minimal metadata: `symbol`, `fiscal_year`, `fiscal_period`, `cik`, `form_type`

**Data Status for AAPL**:
```
7 quarters with EMPTY metrics_data = {}
2024-FY: {} ❌
2024-Q3: {} ❌
2024-Q2: {} ❌
```

**Primary Key**: `(symbol, fiscal_year, fiscal_period)`
**Populated By**: `RDBMSCacheHandler._write()` (src/investigator/infrastructure/cache/rdbms_cache_handler.py:404)
**Problem**: Line 410 defaults to empty dict: `metrics_data=value.get("metrics", {})`

---

## Root Cause Analysis

### Why `quarterly_metrics` is Empty

**File**: `src/investigator/infrastructure/cache/rdbms_cache_handler.py:404-412`

```python
return self.dao.save_metrics(
    symbol=symbol,
    fiscal_year=key_dict.get("fiscal_year"),
    fiscal_period=key_dict.get("fiscal_period"),
    cik=cik,
    form_type=key_dict.get("form_type", "10-K"),
    metrics_data=value.get("metrics", {}),  # ← RETURNS {} IF KEY MISSING
    company_name=value.get("company_name", ""),
)
```

**The Issue**:
1. Cache handler expects `value` dict with `"metrics"` key
2. Caller passes `value` dict WITHOUT `"metrics"` key
3. Defaults to `{}` (empty dict)
4. Database saves empty JSONB

### Why This Matters for DCF

**File**: `utils/dcf_valuation.py` and `agents/fundamental_agent.py`

```python
# fundamental_agent.py tries to get quarterly_data
quarterly_data = self._get_quarterly_data(ticker, max_periods=8)

# If quarterly_metrics is empty, DCF gets no data
# Result: $0.0M FCF calculation
```

---

## Architecture Comparison

| Feature | `sec_companyfacts_processed` | `quarterly_metrics` |
|---------|------------------------------|---------------------|
| **Data Completeness** | ✅ 66 quarters with full data | ❌ 7 quarters with empty `{}` |
| **Schema** | ✅ Explicit 42 columns | ❌ JSONB blob (flexible but opaque) |
| **Ratios** | ✅ Pre-calculated | ❌ Must calculate on read |
| **Quality Metadata** | ✅ Has `data_quality_score` | ❌ No quality tracking |
| **Lineage** | ✅ Links to `raw_data_id` | ❌ No raw data link |
| **Query Performance** | ✅ Indexed columns | ⚠️ JSONB requires GIN index |
| **Type Safety** | ✅ Numeric columns validated | ❌ JSONB can store anything |
| **Maintenance** | ✅ Populated by `SECDataProcessor` | ❌ Populated by cache handler (buggy) |

---

## Recommendation: **Consolidate Around `sec_companyfacts_processed`**

### Why `sec_companyfacts_processed` Should Be the Single Source of Truth

1. **Has ALL the data already** (66 quarters vs 7 empty quarters)
2. **Better schema design** (explicit columns vs JSONB blob)
3. **Pre-calculated ratios** (no runtime calculation overhead)
4. **Quality tracking** (data_quality_score for filtering)
5. **Full lineage** (links back to raw SEC data)
6. **Actively maintained** (`SECDataProcessor` vs buggy cache handler)
7. **Future-proof** (can add columns vs JSONB schema evolution)

### Migration Plan

**Phase 1: Update DCF/GGM to Use `sec_companyfacts_processed`**
```python
# In fundamental_agent.py
def _get_quarterly_data(self, ticker: str, max_periods: int = 8):
    """Get quarterly data from sec_companyfacts_processed table"""
    from utils.db import get_db_manager

    with get_db_manager().get_session() as session:
        results = session.execute(text("""
            SELECT
                fiscal_year,
                fiscal_period,
                operating_cash_flow,
                capital_expenditures,
                free_cash_flow,
                total_revenue,
                net_income,
                /* all other metrics */
            FROM sec_companyfacts_processed
            WHERE symbol = :symbol
            ORDER BY fiscal_year DESC, fiscal_period DESC
            LIMIT :limit
        """), {'symbol': ticker, 'limit': max_periods}).fetchall()

        # Convert to QuarterlyData objects
        return self._convert_to_quarterly_data(results)
```

**Phase 2: Deprecate `quarterly_metrics` Table**
1. Mark as deprecated in code comments
2. Stop writing to it (remove from RDBMSCacheHandler)
3. Monitor for 30 days to ensure no dependencies
4. Drop table after validation

**Phase 3: Add YTD Metadata Column** (if needed)
```sql
ALTER TABLE sec_companyfacts_processed
ADD COLUMN is_ytd_value BOOLEAN DEFAULT FALSE,
ADD COLUMN value_type VARCHAR(20) DEFAULT 'QUARTER'; -- 'QUARTER', 'YTD', 'FY'
```

---

## YTD vs Quarter-Specific Values

### Problem: SEC 10-Q Filings Report YTD (Year-To-Date) Cumulative Values

From raw SEC data analysis:
```
2025-Q3 (period ending 2025-06-28): OCF=$81.75B  ← YTD (Oct 2024 - Jun 2025, 9 months)
2025-Q2 (period ending 2025-03-29): OCF=$53.89B  ← YTD (Oct 2024 - Mar 2025, 6 months)
2025-Q1 (period ending 2024-12-28): OCF=$29.94B  ← Quarter-specific (Oct-Dec 2024, 3 months)
```

### Solution: Calculate Individual Quarters

**For Q2 and Q3, subtract previous quarter**:
```
Q2 individual = Q2 YTD - Q1
Q3 individual = Q3 YTD - Q2 YTD
```

**For Q4, use 10-K (FY) minus Q1+Q2+Q3**:
```
Q4 individual = FY annual - (Q1 + Q2 YTD_to_individual + Q3 YTD_to_individual)
```

### Implementation in `sec_data_processor.py`

Add a post-processing step to convert YTD to individual quarters:

```python
def _convert_ytd_to_individual_quarters(self, processed_filings: List[Dict]) -> List[Dict]:
    """
    Convert YTD cumulative values to individual quarter values

    For cash flow metrics (OCF, CapEx), 10-Q filings report YTD cumulative:
    - Q1: Individual quarter (Oct-Dec)
    - Q2: YTD (Oct-Mar) → need to subtract Q1
    - Q3: YTD (Oct-Jun) → need to subtract Q2 YTD
    """
    # Group by fiscal year
    by_year = {}
    for filing in processed_filings:
        fy = filing['fiscal_year']
        if fy not in by_year:
            by_year[fy] = {}
        by_year[fy][filing['fiscal_period']] = filing

    # Convert YTD to individual for each year
    ytd_metrics = ['operating_cash_flow', 'capital_expenditures', 'free_cash_flow']

    for fy, quarters in by_year.items():
        if 'Q2' in quarters and 'Q1' in quarters:
            for metric in ytd_metrics:
                q2_ytd = quarters['Q2']['data'].get(metric, 0)
                q1_val = quarters['Q1']['data'].get(metric, 0)
                quarters['Q2']['data'][metric] = q2_ytd - q1_val
                quarters['Q2']['data'][f'{metric}_is_ytd'] = False

        if 'Q3' in quarters and 'Q2' in quarters:
            for metric in ytd_metrics:
                q3_ytd = quarters['Q3']['data'].get(metric, 0)
                q2_ytd_original = quarters['Q2']['data'].get(metric, 0) + quarters['Q1']['data'].get(metric, 0)
                quarters['Q3']['data'][metric] = q3_ytd - q2_ytd_original
                quarters['Q3']['data'][f'{metric}_is_ytd'] = False

    return processed_filings
```

---

## Action Items

### Immediate (This Sprint)
- [ ] **Update `fundamental_agent.py`** to read from `sec_companyfacts_processed`
- [ ] **Add debug logging** to trace why `quarterly_metrics` gets empty data
- [ ] **Test DCF/GGM** with `sec_companyfacts_processed` data source

### Short-term (Next Sprint)
- [ ] **Add YTD conversion logic** to `sec_data_processor.py`
- [ ] **Add metadata column** `value_type` to track YTD vs individual quarters
- [ ] **Deprecate `quarterly_metrics`** table with migration plan

### Long-term (Technical Debt)
- [ ] **Remove `quarterly_metrics`** table entirely
- [ ] **Consolidate cache layer** to use only `sec_companyfacts_processed`
- [ ] **Update documentation** to reflect single source of truth

---

## Testing Checklist

Before consolidation:
- [ ] Verify `sec_companyfacts_processed` has data for all test symbols (AAPL, MSFT, GOOGL)
- [ ] Compare DCF results using both tables (should match after YTD conversion)
- [ ] Test edge cases: companies with missing quarters, different fiscal year ends
- [ ] Validate YTD→individual quarter conversion accuracy

After consolidation:
- [ ] Run full regression test suite
- [ ] Verify DCF/GGM calculations return non-zero values
- [ ] Check cache hit rates haven't degraded
- [ ] Monitor database query performance

---

## Files to Modify

### High Priority
1. `agents/fundamental_agent.py` - Update `_get_quarterly_data()` to use `sec_companyfacts_processed`
2. `utils/dcf_valuation.py` - Add debug logging, verify data source
3. `utils/gordon_growth_model.py` - Same as DCF
4. `utils/quarterly_calculator.py` - Add YTD→individual quarter conversion

### Medium Priority
5. `utils/sec_data_processor.py` - Add YTD conversion post-processing
6. `src/investigator/infrastructure/cache/rdbms_cache_handler.py` - Fix or remove `quarterly_metrics` logic

### Low Priority (Cleanup)
7. `utils/db.py` - Deprecate `QuarterlyMetricsDAO` class
8. Database migration script to drop `quarterly_metrics` table

---

## Estimated Impact

**Performance**: ✅ **Improved** (no JSONB parsing, pre-calculated ratios)
**Reliability**: ✅ **Much Better** (populated data vs empty records)
**Maintainability**: ✅ **Simplified** (one table vs two)
**Code Complexity**: ✅ **Reduced** (remove buggy cache handler logic)

**Risk**: ⚠️ **Low** (data already exists in target table, just change query source)

---

## Conclusion

**The `sec_companyfacts_processed` table is the correct, well-designed, actively maintained source of quarterly data.** The `quarterly_metrics` table is a legacy remnant that:
1. Has empty/incomplete data
2. Uses inefficient JSONB storage
3. Lacks quality metadata and lineage
4. Is populated by buggy cache handler code

**Recommendation**: Immediately switch DCF/GGM to use `sec_companyfacts_processed`, then deprecate and eventually remove `quarterly_metrics`.
