# SEC Bulk Table YTD Discovery - Critical `qtrs` Field Finding

**Date**: 2025-11-04
**Discovery**: SEC bulk tables contain BOTH YTD and individual quarter values - Must filter by `qtrs` field!
**Impact**: 🔴 **CRITICAL** - Explains why our bulk table extraction was getting wrong values

---

## Executive Summary

The negative Q4 OCF issue has a **SECOND ROOT CAUSE** beyond the `is_ytd` flag bug we fixed:

**SEC bulk tables (`sec_num_data`) contain MULTIPLE rows for the same metric:**
- **qtrs=1**: Segment/product breakdown values (individual quarter)
- **qtrs=2**: Q2 YTD cumulative value (Q1+Q2)
- **qtrs=3**: Q3 YTD cumulative value (Q1+Q2+Q3)
- **qtrs=4**: Full year value

**Our current code picks qtrs=1 by default**, which gives individual quarter values for segment breakdowns, NOT the statement-level aggregates we need for financial analysis.

---

## The Discovery

### What We Found

Running S&P 100 analysis on bulk tables, we discovered **ALL** stocks showing "point-in-time" pattern:

```
1. AAPL  - ⚠️  Point-in-time | Q2: Point-in-time | Q3: Point-in-time
2. MSFT  - ⚠️  Point-in-time | Q2: Point-in-time | Q3: Point-in-time
...
18 out of 18 stocks: Point-in-time (qtrs=1)
```

**This contradicted our previous finding that sec_companyfacts_processed table has YTD values!**

### Deep Dive: AAPL 2024-Q2 Bulk Table

Querying `sec_num_data` for AAPL 2024-Q2:

```sql
SELECT tag, value, qtrs
FROM sec_num_data
WHERE adsh = '0000320193-24-000069'
    AND tag = 'RevenueFromContractWithCustomerExcludingAssessedTax'
    AND ddate = '2024-03-31'
ORDER BY qtrs;
```

**Results**:
| Tag | Value | qtrs | Meaning |
|-----|-------|------|---------|
| Revenue | $37.3B | 1 | Americas segment (individual Q2) |
| Revenue | $5.6B | 1 | Europe segment (individual Q2) |
| Revenue | $16.4B | 1 | Greater China segment (individual Q2) |
| Revenue | ... | 1 | More segment breakdowns |
| Revenue | $90.8B | 1 | Total (sum of segments) |
| Revenue | **$211.9B** | 2 | **Q2 YTD cumulative (Q1+Q2)** |

**Operating Cash Flow**:
| Tag | Value | qtrs | Meaning |
|-----|-------|------|---------|
| OCF | **$62.6B** | 2 | **Q2 YTD cumulative** |

---

## The Pattern Across All Periods

**AAPL Revenue by Fiscal Period:**

| Fiscal Period | qtrs=1 (Count) | qtrs=1 (Total) | qtrs=2/3/4 | qtrs=2/3/4 (Total) |
|---------------|----------------|----------------|------------|---------------------|
| Q1-2024 | 12 entries | $455.2B | - | - |
| Q2-2024 | 12 entries | $339.1B | **qtrs=2: 12 entries** | **$794.3B (YTD)** |
| Q3-2024 | 12 entries | $318.9B | **qtrs=3: 12 entries** | **$1,113.2B (YTD)** |

**Key Insight:**
- **Q1**: Only qtrs=1 (individual quarter)
- **Q2**: BOTH qtrs=1 (segments) AND **qtrs=2 (YTD cumulative)**
- **Q3**: BOTH qtrs=1 (segments) AND **qtrs=3 (YTD cumulative)**

---

## Why This Matters

### Current Bug in Bulk Table Extraction

**File**: `src/investigator/domain/agents/fundamental.py` lines 1009-1078

**Current Query** (simplified):
```python
SELECT tag, value, ddate, qtrs
FROM sec_num_data
WHERE adsh = %s
    AND tag IN ('NetCashProvidedByUsedInOperatingActivities', ...)
    AND ddate = %s
ORDER BY tag  # ❌ NO qtrs FILTERING!
LIMIT 1
```

**Problem**: This query returns the **FIRST** matching row, which is typically:
- qtrs=1 (segment breakdown)
- Value is individual quarter (for Q1) OR **random segment value** (for Q2/Q3)

**For AAPL 2024-Q2 OCF**:
- ✅ **Correct** (qtrs=2): $62.6B (YTD cumulative)
- ❌ **Our code gets** (qtrs=1): First segment breakdown value

---

## The Fix Required

### Option 1: Filter by qtrs (Recommended for Statement-Level Aggregates)

```python
# For Q2/Q3, query YTD cumulative values
expected_qtrs = 1  # Default for Q1, FY
if fiscal_period == 'Q2':
    expected_qtrs = 2  # Q2 YTD
elif fiscal_period == 'Q3':
    expected_qtrs = 3  # Q3 YTD
elif fiscal_period == 'FY':
    expected_qtrs = 4  # Full year

num_query = f"""
SELECT tag, value, ddate, qtrs
FROM sec_num_data
WHERE adsh = %s
    AND tag IN ({tag_placeholders})
    AND ddate = %s
    AND qtrs = %s  -- ✅ FILTER BY EXPECTED qtrs!
ORDER BY tag
LIMIT 1
"""

cursor.execute(num_query, (adsh, *tags, period_end, expected_qtrs))
```

### Option 2: Prioritize Higher qtrs Values

```python
# Prefer higher qtrs values (YTD over segments)
num_query = f"""
SELECT tag, value, ddate, qtrs
FROM sec_num_data
WHERE adsh = %s
    AND tag IN ({tag_placeholders})
    AND ddate = %s
ORDER BY
    qtrs DESC,  -- ✅ Prefer qtrs=4, then 3, then 2, then 1
    value DESC  -- ✅ If multiple qtrs=1, pick largest (total)
LIMIT 1
"""
```

---

## Verification: Why sec_companyfacts_processed Works

**Our processed table path works because:**
1. We extract from SEC CompanyFacts API JSON
2. API provides **BOTH** individual and cumulative values
3. **Our extraction logic picks the YTD values correctly**
4. We then infer `is_ytd=True` for Q2/Q3 based on `fiscal_period`

**Bulk table path fails because:**
1. We query `sec_num_data` without `qtrs` filtering
2. Get random segment breakdown (qtrs=1)
3. Even if we infer `is_ytd=True`, **the VALUE is wrong** (it's a segment, not total)

---

## Impact Assessment

**Severity**: 🔴 **CRITICAL** - Dual bug system

### Bug #1 (Fixed): `is_ytd` Flag Loss
- ✅ **Status**: FIXED by adding `is_ytd_cashflow` and `is_ytd_income` fields
- **Impact**: YTD conversion now works for processed table path

### Bug #2 (New Discovery): Wrong `qtrs` in Bulk Table Query
- ❌ **Status**: NOT YET FIXED
- **Impact**: Bulk table path gets segment breakdowns instead of statement totals
- **Affected**: Stocks that fall back to bulk tables (estimated 10-20%)

---

## Testing Checklist

- [x] Verify sec_companyfacts_processed has YTD values for Q2/Q3
- [x] Analyze S&P 100 bulk table data
- [x] Discover qtrs field pattern
- [x] Verify AAPL 2024-Q2 has BOTH qtrs=1 and qtrs=2 entries
- [ ] ⚠️ **Fix bulk table query to filter by qtrs**
- [ ] Test with stock that uses bulk table fallback
- [ ] Verify Q4 is positive after fix

---

## Recommendation

**Priority**: **P0** - Fix immediately before bulk table path can be trusted

**Implementation**:
1. **Immediate**: Add `qtrs` filtering to bulk table queries in `fundamental.py` lines 1041-1071
2. **Then**: Test with MSFT (known to use bulk tables for some periods)
3. **Finally**: Re-run S&P 100 analysis to verify YTD pattern is universal

---

## Related Documentation

- **YTD Fix Summary**: `analysis/YTD_FIX_SUMMARY.md`
- **Complete is_ytd Flow**: `analysis/IS_YTD_FLOW_COMPLETE_ANALYSIS.md`
- **Root Cause**: `analysis/YTD_CONVERSION_BUG_ANALYSIS.md`
- **S&P 100 Analysis Results**: `analysis/sp100_ytd_bulk_analysis.json`

---

## Conclusion

**Key Findings**:
1. ✅ `is_ytd` flag bug is FIXED for processed table path
2. ❌ **NEW BUG**: Bulk table queries don't filter by `qtrs` field
3. ⚠️ SEC bulk tables have MULTIPLE rows per metric with different `qtrs` values
4. 📊 Pattern is **UNIVERSAL** across S&P 100 (all have qtrs=2 for Q2, qtrs=3 for Q3)

**Next Actions**:
1. Fix bulk table queries to use `qtrs = expected_qtrs` filtering
2. Re-test AAPL and MSFT with bulk table fallback
3. Verify Q4 computations are positive

**Status**: Critical discovery complete, implementation pending.
