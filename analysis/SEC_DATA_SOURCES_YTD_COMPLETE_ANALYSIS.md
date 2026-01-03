# Complete SEC Data Sources YTD Analysis

**Date**: 2025-11-04
**Finding**: ALL THREE SEC data sources contain YTD values, but with different structures
**Key Insight**: Must filter by duration/qtrs to get correct values

---

## Executive Summary

We've analyzed all three SEC data sources (CompanyFacts API JSON, Bulk Tables, Processed Table) to understand how YTD values are stored and retrieved. **All sources contain YTD data**, but the access pattern differs:

| Data Source | YTD Indicator | How to Get YTD Value |
|-------------|---------------|----------------------|
| **CompanyFacts API** | `start` date | Pick entry with `start = fiscal_year_start` |
| **Bulk Tables** (`sec_num_data`) | `qtrs` field | Filter by `qtrs=2` (Q2), `qtrs=3` (Q3) |
| **Processed Table** | Inferred from `fiscal_period` | Correctly extracted already |

---

## Data Source #1: SEC CompanyFacts API (Raw JSON)

### API Endpoint
```
https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json
```

### JSON Structure
```json
{
  "facts": {
    "us-gaap": {
      "RevenueFromContractWithCustomerExcludingAssessedTax": {
        "units": {
          "USD": [
            {
              "fp": "Q2",
              "fy": 2024,
              "start": "2023-10-01",  ← Fiscal year start
              "end": "2024-03-30",
              "val": 210328000000,    ← Q2 YTD (6 months)
              "form": "10-Q",
              "filed": "2024-05-03"
            },
            {
              "fp": "Q2",
              "fy": 2024,
              "start": "2023-12-31",  ← Q2 start (not fiscal year)
              "end": "2024-03-30",
              "val": 90753000000,     ← Q2 individual (3 months)
              "form": "10-Q",
              "filed": "2024-05-03"
            }
          ]
        }
      }
    }
  }
}
```

### YTD Detection Logic

**Rule**: YTD entry has `start` date == fiscal year start date

```python
# For AAPL with fiscal year Oct 1 - Sep 30
fiscal_year_start = "2023-10-01"  # FY2024 starts Oct 1, 2023

# Filter entries for Q2-2024
q2_entries = [e for e in entries if e['fp'] == 'Q2' and e['fy'] == 2024]

# Pick YTD entry (starts from fiscal year beginning)
ytd_entry = [e for e in q2_entries if e['start'] == fiscal_year_start][0]
# → val: $210.3B (YTD cumulative)

# Individual quarter entry (starts from quarter beginning)
individual_entry = [e for e in q2_entries if e['start'] != fiscal_year_start][0]
# → val: $90.8B (Q2 individual)
```

### Complete Pattern for All Quarters

**AAPL 2024 Revenue:**

| Period | Entry Type | Start Date | End Date | Value | Duration |
|--------|-----------|------------|----------|-------|----------|
| Q1-2024 | Individual | 2023-10-01 | 2023-12-30 | $119.6B | 3 months |
| Q2-2024 | **YTD** | **2023-10-01** | 2024-03-30 | **$210.3B** | **6 months** |
| Q2-2024 | Individual | 2023-12-31 | 2024-03-30 | $90.8B | 3 months |
| Q3-2024 | **YTD** | **2023-10-01** | 2024-06-29 | **$296.1B** | **9 months** |
| Q3-2024 | Individual | 2024-03-31 | 2024-06-29 | $85.8B | 3 months |
| FY-2024 | Full Year | 2023-10-01 | 2024-09-28 | $391.0B | 12 months |

**Pattern Observed:**
- Q1: Only 1 entry (individual quarter)
- Q2: **2 entries** - YTD (6 mo) + individual (3 mo)
- Q3: **2 entries** - YTD (9 mo) + individual (3 mo)
- FY: Only 1 entry (full year)

---

## Data Source #2: SEC Bulk Tables (PostgreSQL)

### Tables
- `sec_sub_data`: Submission metadata (adsh, cik, fy, fp, form, period, filed)
- `sec_num_data`: Numeric XBRL tag values (adsh, tag, value, **qtrs**, ddate)

### Key Field: `qtrs` (Duration in Quarters)

**Definition**: Number of quarters covered by the value
- `qtrs=0`: Point-in-time snapshot (balance sheet items)
- `qtrs=1`: 1 quarter duration (individual Q1, or segment breakdowns)
- `qtrs=2`: 2 quarters duration (**Q2 YTD** = Q1+Q2)
- `qtrs=3`: 3 quarters duration (**Q3 YTD** = Q1+Q2+Q3)
- `qtrs=4`: 4 quarters duration (full year)

### SQL Query Example

```sql
-- Get AAPL Q2-2024 Operating Cash Flow (YTD)
SELECT 
    s.fy, s.fp, n.tag, n.value, n.qtrs
FROM sec_sub_data s
JOIN sec_num_data n ON s.adsh = n.adsh
WHERE s.cik = '0000320193'
    AND s.fy = 2024
    AND s.fp = 'Q2'
    AND n.tag = 'NetCashProvidedByUsedInOperatingActivities'
    AND n.ddate = s.period
    AND n.qtrs = 2  -- ✅ Filter for Q2 YTD!
```

**Result**: $62.585B (Q2 YTD cumulative)

### Complete Pattern for AAPL 2024

**Operating Cash Flow:**

| Period | qtrs | Value | Type |
|--------|------|-------|------|
| Q1-2024 | 1 | $34.01B | Individual quarter |
| Q2-2024 | **2** | **$62.58B** | **YTD cumulative (Q1+Q2)** |
| Q3-2024 | **3** | **$88.94B** | **YTD cumulative (Q1+Q2+Q3)** |
| FY-2024 | 4 | $122.15B | Full year |

**Revenue (qtrs distribution):**

| Period | qtrs=1 (Count) | qtrs=1 (Total) | qtrs=2/3/4 | qtrs=2/3/4 Value |
|--------|----------------|----------------|------------|------------------|
| Q1-2024 | 12 segment entries | $455.2B (sum) | - | - |
| Q2-2024 | 12 segment entries | $339.1B (sum) | **qtrs=2: 12 entries** | **$794.3B (YTD)** |
| Q3-2024 | 12 segment entries | $318.9B (sum) | **qtrs=3: 12 entries** | **$1,113.2B (YTD)** |

**Key Insight**: For revenue, bulk tables have:
- **qtrs=1**: Segment/product breakdowns (Americas, Europe, etc.) - individual quarters
- **qtrs=2/3**: Statement-level totals - YTD cumulative

**Current Bug**: Our code doesn't filter by `qtrs`, so it picks first matching row (usually qtrs=1 segment breakdown)!

---

## Data Source #3: sec_companyfacts_processed Table

### Schema (Flat Columns)
```sql
CREATE TABLE sec_companyfacts_processed (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    fiscal_year INTEGER,
    fiscal_period VARCHAR(5),  -- Q1, Q2, Q3, FY
    operating_cash_flow NUMERIC(20,2),
    total_revenue NUMERIC(20,2),
    -- ... more flat columns
    -- NO is_ytd COLUMN!
);
```

### Extraction Process

**Source**: Processes CompanyFacts API JSON → Flattens to relational table

**Logic** (in extraction script):
1. Fetch CompanyFacts API JSON
2. For each metric, filter entries by `fiscal_period`
3. **Pick entry with longest duration** (start == fiscal_year_start)
4. Store in flat table with `fiscal_year`, `fiscal_period`, and metric values

### YTD Inference (At Query Time)

**File**: `src/investigator/domain/agents/fundamental.py` line 1206

```python
# Query returns row from sec_companyfacts_processed
result = db.execute(query).fetchone()

# Infer is_ytd from fiscal_period
is_ytd = fiscal_period in ['Q2', 'Q3']  # ✅ CORRECT

# Build statement-level structure
data = {
    "cash_flow": {
        "operating_cash_flow": to_float(result.operating_cash_flow),
        "is_ytd": is_ytd  # ✅ FLAG SET
    },
    "income_statement": {
        "total_revenue": to_float(result.total_revenue),
        "is_ytd": is_ytd  # ✅ FLAG SET
    }
}
```

**Why This Works:**
1. Processed table stores **YTD values** (extracted correctly from API JSON)
2. We infer `is_ytd=True` for Q2/Q3 at query time
3. Flag is preserved through QuarterlyData construction
4. YTD conversion function sees `is_ytd: True` and performs conversion

---

## Comparison Table

| Aspect | CompanyFacts API | Bulk Tables | Processed Table |
|--------|------------------|-------------|-----------------|
| **YTD Storage** | Multiple entries per period | Multiple rows with different `qtrs` | Single row (pre-filtered) |
| **YTD Indicator** | `start` date (fiscal year start) | `qtrs` field (2 for Q2, 3 for Q3) | Inferred from `fiscal_period` |
| **Selection Logic** | Pick entry where `start == fy_start` | Filter by `qtrs=2` or `qtrs=3` | Already correct (longest duration) |
| **Current Status** | ✅ Works (processed table extraction) | ❌ **BUG**: No `qtrs` filtering | ✅ Works (correct inference) |

---

## The Bugs We Found

### Bug #1: `is_ytd` Flag Loss (FIXED)
- **Location**: `fundamental.py` lines 95-110 (`QuarterlyData.to_dict()`)
- **Problem**: Hardcoded `is_ytd: False` for all quarters
- **Fix**: Added `is_ytd_cashflow` and `is_ytd_income` fields to QuarterlyData
- **Status**: ✅ FIXED for processed table path

### Bug #2: Bulk Table Missing `qtrs` Filter (NOT FIXED)
- **Location**: `fundamental.py` lines 1041-1071 (bulk table fallback)
- **Problem**: Queries `sec_num_data` without filtering by `qtrs`
- **Impact**: Gets random segment breakdown (qtrs=1) instead of statement total (qtrs=2/3)
- **Fix Needed**: Add `qtrs` filtering:
  ```python
  expected_qtrs = 1 if fiscal_period == 'Q1' else 2 if fiscal_period == 'Q2' else 3 if fiscal_period == 'Q3' else 4
  ```
- **Status**: ❌ NOT YET FIXED

---

## Recommendations

**Priority**: **P0** - Critical for data accuracy

### Immediate Actions

1. **Fix Bulk Table Queries**:
   - Add `qtrs` filtering to all `sec_num_data` queries
   - Test with stocks that use bulk table fallback

2. **Verify S&P 100 Pattern**:
   - Re-run analysis script with `qtrs` filtering
   - Confirm 100% of stocks have qtrs=2 for Q2, qtrs=3 for Q3

3. **Update Documentation**:
   - Document `qtrs` field importance
   - Add query examples with correct filtering

### Long-Term Improvements

1. **Standardize on Processed Table**:
   - Backfill historical data into `sec_companyfacts_processed`
   - Minimize reliance on bulk table fallback

2. **Add Validation**:
   - Check if Q2/Q3 values make sense (should be > Q1)
   - Flag anomalies for manual review

3. **Automated Testing**:
   - Test YTD conversion with known good values
   - Verify Q4 computation produces positive values

---

## Conclusion

**Key Findings:**
1. ✅ All three SEC data sources contain YTD values
2. ✅ CompanyFacts API: Filter by `start` date
3. ❌ Bulk tables: **MUST** filter by `qtrs` field (currently missing!)
4. ✅ Processed table: Pre-filtered correctly, just need `is_ytd` inference

**Next Actions:**
1. Implement `qtrs` filtering in bulk table queries
2. Test with MSFT or other stocks using bulk fallback
3. Verify Q4 computations are positive across S&P 100

**Status**: Critical analysis complete, bulk table fix pending implementation.

---

## Related Documentation

- **Bulk Table Discovery**: `analysis/BULK_TABLE_YTD_DISCOVERY.md`
- **YTD Fix Summary**: `analysis/YTD_FIX_SUMMARY.md`
- **is_ytd Flow**: `analysis/IS_YTD_FLOW_COMPLETE_ANALYSIS.md`
- **Root Cause**: `analysis/YTD_CONVERSION_BUG_ANALYSIS.md`
