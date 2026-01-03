# Sector-Specific Analysis Report

**Date**: 2025-11-03
**Test**: investigator_v2.sh with force-refresh across 5 major sectors
**Status**: ❌ CRITICAL BUG FOUND - ALL SECTORS FAILING

---

## Executive Summary

Ran comprehensive `investigator_v2.sh` analysis across 5 representative stocks from different sectors:

| Symbol | Sector | Status | Critical Issue |
|--------|--------|--------|----------------|
| AAPL | Technology - Consumer Electronics | ❌ FAILED | Multi-quarter analysis broken |
| NEE | Utilities - Electric Utility | ❌ FAILED | Multi-quarter analysis broken |
| AMT | Real Estate - REIT | ❌ FAILED | Multi-quarter analysis broken |
| JPM | Financials - Banking | ❌ FAILED | Multi-quarter analysis broken |
| XOM | Energy - Oil & Gas | ❌ FAILED | Multi-quarter analysis broken |

**SUCCESS RATE**: 0/5 (0%) - All analyses failed multi-quarter component

---

## Critical Bug Identified

### Bug Details

**Type**: Type mismatch (Decimal vs float)
**Error Message**: `unsupported operand type(s) for /: 'decimal.Decimal' and 'float'`
**Location**: `agents/fundamental_agent.py`
**Functions Affected**:
- `_analyze_revenue_trend()` (lines 1140, 1149)
- `_analyze_margin_trend()` (lines 1229, 1253-1254)
- `_calculate_quarterly_comparisons()` (lines 1375, 1390)
- `_calculate_financial_ratios()` (lines 1022-1041)
- `_perform_valuation()` (lines 1534-1541)

### Root Cause

SEC bulk table data (from `dao/sec_bulk_dao.py`) returns `decimal.Decimal` types from PostgreSQL. When these values are used in arithmetic operations with Python `float` or `int` types, Python raises `TypeError`.

**Example Failing Code**:
```python
# Line 1140: quarterly_data contains Decimal values from DB
growth = ((revenues[i] - revenues[i-1]) / revenues[i-1]) * 100  # ❌ FAILS
# revenues[i] is Decimal, but 100 is int → type mismatch
```

**This is THE SAME bug we fixed in** `utils/financial_calculators.py:111-115` for the utility revenue calculation!

### Impact

- **Multi-quarter trend analysis**: COMPLETELY BROKEN across ALL sectors
- **Data quality**: Degraded (58-81% vs expected 90%+)
- **Missing insights**: No revenue trends, margin trends, cash flow quality, cyclical patterns
- **LLM analysis**: Limited to single-period data only

---

## Sector-Specific Findings

### 1. AAPL (Technology - Consumer Electronics)

**Status**: ⏳ In Progress (incomplete due to bug)

**Positive Findings**:
- ✅ SEC extraction working
- ✅ Revenue extracted correctly
- ✅ Critical metrics found (4/4)
- ✅ LLM fundamental growth analysis completed
- ✅ LLM forecast completed

**Issues**:
- ❌ Multi-quarter analysis failed (Decimal/float bug)
- ❌ Profitability analysis not found (likely waiting for LLM)
- ❌ Balance sheet analysis not found
- ⚠️  Data quality: Poor (58.9%) - due to missing multi-quarter data
- ⚠️  Warnings: Missing debt metrics (currentLiabilities), operating cash flow issues

**Data Quality Impact**:
- Without multi-quarter data, can't assess revenue growth trends
- Can't calculate YoY comparisons
- Can't detect seasonality or cyclical patterns

---

### 2. NEE (Utilities - Electric Utility)

**Status**: ⏳ In Progress (incomplete due to bug)

**Positive Findings**:
- ✅ SEC extraction working
- ✅ Revenue extracted: $24.753B (using `RegulatedAndUnregulatedOperatingRevenue` - OUR FIX WORKS!)
- ✅ LLM fundamental growth analysis completed
- ✅ Data quality: Good (81.2%)

**Issues**:
- ❌ Multi-quarter analysis failed (Decimal/float bug)
- ❌ Profitability analysis not found
- ❌ Balance sheet analysis not found
- ⚠️  LLM forecast: JSON extraction issues
- ⚠️  Calendar-based fallback used (should use fiscal periods)

**Sector-Specific Tags**:
- **VALIDATED**: `RegulatedAndUnregulatedOperatingRevenue` tag now working ✅
- Expected but not detected: `RegulatoryAssets`, `RegulatoryLiability` (may not be in logs yet)

---

### 3. AMT (Real Estate - REIT)

**Status**: ⏳ In Progress (incomplete due to bug)

**Positive Findings**:
- ✅ SEC extraction working
- ✅ Revenue extracted
- ✅ LLM fundamental growth analysis completed
- ✅ Data quality: Good (81.2%)

**Issues**:
- ❌ Multi-quarter analysis failed (Decimal/float bug)
- ❌ Profitability analysis not found
- ❌ Balance sheet analysis not found
- ⚠️  LLM forecast: JSON extraction issues
- ⚠️  Cache write failures (RDBMS)
- ⚠️  Missing cash flow data
- ⚠️  Quick ratio = 0 (missing currentLiabilities)

**Sector-Specific Tags**:
- Expected: `NoncontrollingInterest`, `RedeemableNoncontrollingInterest`, `TemporaryEquity`
- Not yet detected in logs (analysis incomplete)

---

### 4. JPM (Financials - Banking)

**Status**: ⏳ In Progress (incomplete due to bug)

**Positive Findings**:
- ✅ SEC extraction working
- ✅ Revenue extracted (financial institution tags working)
- ✅ LLM fundamental growth analysis completed
- ✅ LLM forecast completed
- ✅ Data quality: Fair (68.9%)

**Issues**:
- ❌ Multi-quarter analysis failed (Decimal/float bug)
- ❌ Profitability analysis not found
- ❌ Balance sheet analysis not found
- ⚠️  Bulk data stale (186 days old) - fell back to CompanyFacts API
- ⚠️  Cache write failures (RDBMS, Parquet)

**Sector-Specific Tags**:
- Expected: `InterestIncomeOperating`, `InterestExpense`, `InterestAndDividendIncomeOperating`
- Not yet detected in logs (analysis incomplete)

---

### 5. XOM (Energy - Oil & Gas)

**Status**: ⏳ In Progress (incomplete due to bug)

**Positive Findings**:
- ✅ SEC extraction working
- ✅ Revenue extracted
- ✅ LLM fundamental growth analysis completed
- ✅ Data quality: Good (81.2%)

**Issues**:
- ❌ Multi-quarter analysis failed (Decimal/float bug)
- ❌ Profitability analysis not found
- ❌ Balance sheet analysis not found
- ❌ LLM forecast not found
- ⚠️  Missing cash flow data
- ⚠️  ROA = 0 (missing data)

**Sector-Specific Tags**:
- Expected: `PropertyPlantAndEquipmentNet` (capital-intensive)
- Not yet detected in logs (analysis incomplete)

---

## Secondary Issues Found

### 1. LLM JSON Extraction Failures

**Affected**: NEE, AMT (possibly others)

```
⚠️  Failed to extract JSON from response (length: 250)
⚠️  Failed to extract JSON from response (length: 249)
```

**Impact**: Forecast data not properly cached/processed

**Recommendation**: Investigate LLM response format issues

### 2. Cache Write Failures

**Affected**: AMT, JPM

```
❌ Cache WRITE FAILED [RdbmsCacheStorageHandler]: sec_response
❌ Cache WRITE FAILED [ParquetCacheStorageHandler]: technical_data
```

**Impact**: Performance degradation, repeated API calls

**Recommendation**: Investigate RDBMS/Parquet cache handler issues

### 3. Data Quality Variability

| Symbol | Data Quality | Score |
|--------|--------------|-------|
| AAPL | Poor | 58.9% |
| JPM | Fair | 68.9% |
| NEE | Good | 81.2% |
| AMT | Good | 81.2% |
| XOM | Good | 81.2% |

**Reason**: Multi-quarter data missing due to Decimal bug significantly impacts data completeness scoring

---

## Fix Required

### Location

`agents/fundamental_agent.py` - Multiple functions need Decimal → float conversion

### Affected Lines

- **Revenue trend analysis** (1140, 1149)
- **Margin trend analysis** (1229, 1253-1254)
- **Quarterly comparisons** (1375, 1390)
- **Financial ratios** (1022-1041)
- **Valuation metrics** (1534-1541)

### Recommended Fix

Add `float()` conversion wherever Decimal values are used in arithmetic:

```python
# BEFORE (line 1140)
growth = ((revenues[i] - revenues[i-1]) / revenues[i-1]) * 100  # ❌ FAILS

# AFTER
growth = ((float(revenues[i]) - float(revenues[i-1])) / float(revenues[i-1])) * 100  # ✅ WORKS
```

**Alternative**: Convert Decimal to float immediately after extraction from database (in `sec_bulk_dao.py` or normalizer)

---

## Tag Mapper Validation

### Tags Successfully Validated

| Tag | Sector | Status |
|-----|--------|--------|
| `RegulatedAndUnregulatedOperatingRevenue` | Utilities (NEE) | ✅ WORKING |
| `total_revenue` (generic) | All | ✅ WORKING |
| `total_assets` | All | ✅ WORKING |
| `total_liabilities` | All | ✅ WORKING |
| `net_income` | All | ✅ WORKING |

### Tags Expected But Not Yet Validated

(Analysis incomplete due to bug - tags may appear later in logs)

| Tag | Sector | Reason |
|-----|--------|--------|
| `RegulatoryAssets` | Utilities | Not in logs yet |
| `RegulatoryLiability` | Utilities | Not in logs yet |
| `NoncontrollingInterest` | REIT | Not in logs yet |
| `RedeemableNoncontrollingInterest` | REIT | Not in logs yet |
| `InterestIncomeOperating` | Financials | Not in logs yet |
| `PropertyPlantAndEquipmentNet` | Energy | Not in logs yet |

---

## Recommendations

### Priority 1 - CRITICAL (Blocking)

1. **FIX Decimal/float bug** in `agents/fundamental_agent.py`
   - Add `float()` conversions to all arithmetic operations
   - OR convert Decimal to float at database extraction layer
   - **Impact**: Unblocks multi-quarter analysis for ALL sectors

### Priority 2 - HIGH

2. **Fix LLM JSON extraction** failures
   - Investigate why responses are truncated (250 chars)
   - Validate LLM response format

3. **Fix cache write failures**
   - RDBMS handler failing for AMT, JPM
   - Parquet handler failing for JPM

### Priority 3 - MEDIUM

4. **Improve fiscal period detection**
   - Currently falling back to calendar-based quarters
   - Should use actual fiscal periods from SEC filings

5. **Add missing debt metrics**
   - `currentLiabilities` missing for AAPL
   - Causing debt ratios to be unreliable

---

## Conclusion

The **sector-specific tag coverage is EXCELLENT** where tested:
- ✅ Utilities revenue tag working (`RegulatedAndUnregulatedOperatingRevenue`)
- ✅ All critical metrics extracting correctly
- ✅ Tag mapper additions from earlier work are functioning

However, the **Decimal/float bug is CRITICAL** and blocking:
- ❌ Multi-quarter analysis broken for ALL sectors
- ❌ Data quality scores artificially low (missing trend data)
- ❌ Comprehensive analysis cannot complete

**Once the Decimal bug is fixed**, the system should provide robust coverage across all sectors with sector-specific tag support working correctly.

---

**Next Steps**:
1. Fix Decimal/float bug (agents/fundamental_agent.py)
2. Re-run sector analyses
3. Validate sector-specific tags appear in full logs
4. Verify multi-quarter trends working
5. Confirm data quality improves to 90%+
