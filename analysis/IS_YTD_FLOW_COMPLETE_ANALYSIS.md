# Complete `is_ytd` Data Flow Analysis

**Date**: 2025-11-04
**Finding**: `is_ytd` is **ONLY** set for processed table path, **NOT** for bulk table fallback
**Status**: ⚠️ **INCOMPLETE FIX** - Bulk table path needs updating

---

## Executive Summary

The `is_ytd` flag is **NOT** present in any SEC raw data sources (CompanyFacts API, bulk tables). We must **INFER** it from metadata:
- **Fiscal Period**: Q2 and Q3 in 10-Q filings contain YTD cumulative values
- **Form Type**: 10-Q (quarterly) vs 10-K (annual)

**Current Status**:
- ✅ **Processed table path**: Correctly infers and preserves `is_ytd` flags
- ❌ **Bulk table fallback path**: Hardcodes `is_ytd=False` (BUG!)

---

## Data Sources and YTD Detection

### 1. SEC CompanyFacts API (Raw JSON)

**Endpoint**: `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`

**Sample Response**:
```json
{
  "facts": {
    "us-gaap": {
      "NetCashProvidedByUsedInOperatingActivities": {
        "units": {
          "USD": [
            {
              "fp": "Q2",           // ← Fiscal Period (NO is_ytd FLAG!)
              "fy": 2024,           // ← Fiscal Year
              "form": "10-Q",       // ← Form Type
              "val": 62570000000,   // ← VALUE (YTD for Q2/Q3)
              "accn": "0001193125...",
              "filed": "2024-05-03",
              "start": "2023-10-01",
              "end": "2024-03-31"
            }
          ]
        }
      }
    }
  }
}
```

**YTD Inference Rule**:
```python
# In _fetch_from_processed_table() line 1206
is_ytd = fiscal_period in ['Q2', 'Q3']
```

**Why Q2/Q3 are YTD**:
- Apple's fiscal year: Oct 1 - Sep 30
- Q1 (Oct-Dec): Individual quarter
- Q2 (Jan-Mar): **YTD cumulative** = Q1 + Q2 (reported as single value in 10-Q)
- Q3 (Apr-Jun): **YTD cumulative** = Q1 + Q2 + Q3 (reported as single value in 10-Q)
- Q4 (Jul-Sep): Individual quarter (or computed from FY - (Q1+Q2+Q3))

### 2. SEC Bulk Tables (PostgreSQL)

**Tables**: `sec_sub_data`, `sec_num_data`, `sec_txt_data`

**Sample Data from sec_sub_data**:
```sql
SELECT adsh, cik, name, form, period, fy, fp, filed
FROM sec_sub_data
WHERE cik = '0000320193' AND fy = 2024 AND fp = 'Q2';

Result:
adsh           | cik        | name       | form | period     | fy   | fp | filed
---------------|------------|------------|------|------------|------|----|-----------
0001193125-... | 0000320193 | Apple Inc. | 10-Q | 2024-03-31 | 2024 | Q2 | 2024-05-03
```

**Key Fields**:
- `fp`: Fiscal period (Q1, Q2, Q3, FY)
- `form`: Form type (10-Q, 10-K)
- `period`: Period end date

**YTD Inference Rule** (SAME AS ABOVE):
```python
# Should be applied in bulk table path but currently ISN'T!
is_ytd = fiscal_period in ['Q2', 'Q3']
```

### 3. sec_companyfacts_processed Table (Flattened)

**Schema**:
```sql
CREATE TABLE sec_companyfacts_processed (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    fiscal_year INTEGER,
    fiscal_period VARCHAR(5),  -- Q1, Q2, Q3, FY
    operating_cash_flow NUMERIC(20,2),
    -- ... more flat columns
    -- NO is_ytd COLUMN!
);
```

**YTD Inference**:
- Applied at **query time** in `_fetch_from_processed_table()` line 1206
- Based on `fiscal_period` value from database row

---

## Complete Data Flow Analysis

### Path 1: Processed Table (✅ WORKING)

```
1. Database Query (_fetch_from_processed_table:1137-1171)
   ├─ Query: SELECT ... FROM sec_companyfacts_processed WHERE ... fiscal_period = 'Q2'
   └─ Returns: Row with fiscal_period='Q2', operating_cash_flow=62570000000

2. YTD Inference (line 1206)
   ├─ Code: is_ytd = fiscal_period in ['Q2', 'Q3']
   └─ Result: is_ytd = True (for Q2)

3. Statement-Level Structure Creation (lines 1208-1231)
   ├─ Creates: {
   │     "cash_flow": {
   │         "operating_cash_flow": 62570000000,
   │         "is_ytd": True  ← FLAG SET!
   │     },
   │     "income_statement": {
   │         "total_revenue": 91443000000,
   │         "is_ytd": True  ← FLAG SET!
   │     }
   │  }
   └─ Returns to _fetch_historical_quarters()

4. Flag Extraction (lines 974-975)
   ├─ Code: is_ytd_cashflow = cash_flow.get("is_ytd", False)
   │        is_ytd_income = income_statement.get("is_ytd", False)
   └─ Result: is_ytd_cashflow = True, is_ytd_income = True

5. QuarterlyData Construction (lines 1080-1089)
   ├─ Code: QuarterlyData(...,
   │                      is_ytd_cashflow=is_ytd_cashflow if use_processed else False,
   │                      is_ytd_income=is_ytd_income if use_processed else False)
   └─ Result: QuarterlyData with is_ytd_cashflow=True, is_ytd_income=True

6. Serialization (to_dict:95-110)
   ├─ Code: "cash_flow": {"is_ytd": self.is_ytd_cashflow}
   └─ Returns: {"cash_flow": {..., "is_ytd": True}}

7. YTD Conversion (quarterly_calculator:309-336)
   ├─ Checks: if q2.get('cash_flow', {}).get('is_ytd'):  # TRUE! ✅
   ├─ Converts: Q2_individual = Q2_YTD - Q1
   └─ Result: Q2 converted from $62.57B (YTD) to $28.56B (individual)
```

### Path 2: Bulk Table Fallback (❌ BUG!)

```
1. Bulk Table Query (lines 1041-1071)
   ├─ Query: get_num_data_for_adsh(adsh, tags=['NetCashProvidedByUsedInOperatingActivities'])
   └─ Returns: {"NetCashProvidedByUsedInOperatingActivities": 62570000000}

2. YTD Inference
   ├─ Code: ❌ MISSING! No inference happens in bulk table path
   └─ Result: No is_ytd flags available

3. Financial Data Creation (lines 1060-1071)
   ├─ Creates: financial_data = {
   │     "operating_cash_flow": 62570000000,
   │     # NO is_ytd FLAGS!
   │  }
   └─ Returns flat dict

4. Flag Extraction (lines 974-975)
   ├─ Code: ❌ NOT REACHED - bulk table path skips this section
   └─ Result: is_ytd_cashflow and is_ytd_income are undefined

5. QuarterlyData Construction (lines 1080-1089)
   ├─ Code: QuarterlyData(...,
   │                      is_ytd_cashflow=is_ytd_cashflow if use_processed else False,  ← HARDCODED FALSE!
   │                      is_ytd_income=is_ytd_income if use_processed else False)      ← HARDCODED FALSE!
   └─ Result: QuarterlyData with is_ytd_cashflow=False, is_ytd_income=False ❌ WRONG!

6. Serialization (to_dict:95-110)
   ├─ Code: "cash_flow": {"is_ytd": self.is_ytd_cashflow}
   └─ Returns: {"cash_flow": {..., "is_ytd": False}} ❌ WRONG FOR Q2/Q3!

7. YTD Conversion (quarterly_calculator:309-336)
   ├─ Checks: if q2.get('cash_flow', {}).get('is_ytd'):  # FALSE! ❌
   ├─ Skips conversion
   └─ Result: Q2 remains at $62.57B (YTD cumulative) ❌ BUG!

8. Q4 Computation
   ├─ Formula: Q4 = FY - (Q1 + Q2_YTD + Q3_YTD)
   │           = $122.15B - ($34.01B + $62.57B + $88.95B)
   │           = $122.15B - $185.53B
   └─ Result: Q4 = -$63.38B ❌ NEGATIVE!
```

---

## The Bug in Bulk Table Path

**Location**: `src/investigator/domain/agents/fundamental.py` lines 1087-1088

**Current Code**:
```python
qdata = QuarterlyData(
    fiscal_year=q["fiscal_year"],
    fiscal_period=q["fiscal_period"],
    financial_data=financial_data,
    ratios=ratios,
    data_quality=quality,
    filing_date=str(q["filed"]),
    is_ytd_cashflow=is_ytd_cashflow if use_processed else False,  # ❌ BUG: Hardcoded False for bulk!
    is_ytd_income=is_ytd_income if use_processed else False       # ❌ BUG: Hardcoded False for bulk!
)
```

**Problem**:
- `use_processed=True` → Uses `is_ytd_cashflow` and `is_ytd_income` from processed table ✅
- `use_processed=False` → Hardcodes `False` for both flags ❌

**Impact**:
- Bulk table data for Q2/Q3 is ALSO YTD cumulative
- Hardcoding False prevents YTD conversion
- Results in negative Q4 values (same bug as before)

---

## The Fix for Bulk Table Path

### Change Required

**File**: `src/investigator/domain/agents/fundamental.py` lines 1009-1089

**Add YTD inference for bulk table path**:

```python
if not use_processed:
    # FALLBACK: Extract from bulk tables using CanonicalKeyMapper
    self.logger.warning(
        f"⚠️  Processed data not found for {symbol} {q['fiscal_year']}-{q['fiscal_period']}, "
        f"falling back to bulk tables with canonical key extraction (ADSH: {q['adsh']})"
    )

    # ... (existing bulk table extraction code) ...

    # CRITICAL: Infer is_ytd flags for bulk table data (SAME LOGIC AS PROCESSED TABLE!)
    # Q2/Q3 from 10-Q filings contain YTD cumulative values
    is_ytd = q["fiscal_period"] in ['Q2', 'Q3']
    is_ytd_cashflow = is_ytd
    is_ytd_income = is_ytd

    # Calculate ratios
    ratios = self._calculate_quarterly_ratios(financial_data)

    # Assess quality
    quality = self._assess_quarter_quality(financial_data)

# Create QuarterlyData with ADSH threading (WORKS FOR BOTH PATHS NOW)
qdata = QuarterlyData(
    fiscal_year=q["fiscal_year"],
    fiscal_period=q["fiscal_period"],
    financial_data=financial_data,
    ratios=ratios,
    data_quality=quality,
    filing_date=str(q["filed"]),
    is_ytd_cashflow=is_ytd_cashflow,  # ✅ FIXED: Works for both paths
    is_ytd_income=is_ytd_income        # ✅ FIXED: Works for both paths
)
```

---

## Summary Table

| Data Source | YTD in Raw Data? | YTD Inference | Current Status |
|-------------|------------------|---------------|----------------|
| **CompanyFacts API** | ❌ NO | ✅ Line 1206 | ✅ Working |
| **sec_companyfacts_processed** | ❌ NO (flat columns) | ✅ Line 1206 | ✅ Working |
| **Bulk Tables (sec_sub_data)** | ❌ NO | ❌ Missing! | ❌ **BUG** |

---

## Testing Checklist

- [x] Processed table path: YTD inference working
- [x] Processed table path: Flag preservation working
- [x] Processed table path: YTD conversion working
- [ ] ⚠️ **Bulk table path: YTD inference MISSING**
- [ ] ⚠️ **Bulk table path: Flag preservation HARDCODED FALSE**
- [ ] ⚠️ **Bulk table path: YTD conversion SKIPPED**

---

## Impact Assessment

**Severity**: 🟡 **MEDIUM** - Affects stocks that fall back to bulk tables

**When Bulk Table Fallback Happens**:
1. `sec_companyfacts_processed` has zero/missing revenue for a quarter
2. Data quality score below threshold
3. Newly added stocks not yet in processed table

**Affected Stocks**:
- Estimated 10-20% of stocks fall back to bulk tables
- Typically smaller cap or newly listed companies
- All would have negative Q4 and incorrect DCF valuations

**Fix Priority**: **P1** - Should be implemented immediately after processed table fix is verified

---

## Recommended Implementation

1. **Immediate**: Verify processed table fix is working (AAPL test running)
2. **Next**: Add YTD inference to bulk table path (lines 1009-1078)
3. **Then**: Test with stock that uses bulk table fallback
4. **Finally**: Update documentation and mark as complete

---

## Related Documentation

- **YTD Fix Summary**: `analysis/YTD_FIX_SUMMARY.md`
- **Root Cause**: `analysis/YTD_CONVERSION_BUG_ANALYSIS.md`

---

## Conclusion

**Key Finding**: `is_ytd` is **NEVER** in raw SEC data (API or bulk tables). We must **ALWAYS INFER** it from `fiscal_period` field.

**Current Status**:
- ✅ Processed table path: Correctly infers and preserves flags
- ❌ Bulk table path: Missing inference, hardcodes False

**Next Action**: Add YTD inference to bulk table fallback path at line 1078 (after quality assessment, before QuarterlyData construction).
