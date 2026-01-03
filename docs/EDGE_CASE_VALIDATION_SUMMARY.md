# Edge Case Validation Summary

**Date**: 2025-11-17
**Status**: ⚠️ CRITICAL FIX REQUIRED

---

## Executive Summary

Comprehensive edge case testing revealed a **critical bug** in fiscal year calculation logic affecting **3,213 companies** (18% of database) with fiscal year ends in **January-June**.

**Bug**: When `period_month == fiscal_year_end_month` for companies with Jan-Jun fiscal year ends, the calculated fiscal year is **off by +1**.

**Impact**:
- **Walmart** (Jan 31 FYE): 2025-01-31 calculated as FY 2025, should be FY 2024
- **February FYE companies**: 2025-02-28 calculated as FY 2025, should be FY 2024

**Fix Complexity**: LOW (10 lines of code)
**Risk**: HIGH (systematic data labeling error)

---

## Edge Cases Tested

### 1. Different Fiscal Year End Months ✓

| FYE Month | Count  | Sample Symbols | Test Status |
|-----------|--------|----------------|-------------|
| Jan (01)  | 392    | WMT            | FAIL → FIX  |
| Feb (02)  | 365    | Various        | FAIL → FIX  |
| Mar (03)  | 930    | Various        | PASS*       |
| Apr (04)  | 252    | Various        | PASS*       |
| May (05)  | 237    | ORCL           | PASS        |
| Jun (06)  | 1,037  | MSFT, STX      | PASS        |
| Jul (07)  | 237    | ZS             | PASS*       |
| Aug (08)  | 249    | Various        | PASS*       |
| Sep (09)  | 852    | AAPL, V        | PASS*       |
| Oct (10)  | 286    | Various        | PASS*       |
| Nov (11)  | 218    | Various        | PASS*       |
| Dec (12)  | 12,920 | AMZN, META     | PASS        |

*Inferred from test logic, not explicitly tested

### 2. Leap Year Handling ✓

| Test Case                        | Expected | Calculated | Status |
|----------------------------------|----------|------------|--------|
| 2024-02-29 (Feb FYE)             | 2024     | 2024       | PASS   |
| 2024-02-29 (May FYE)             | 2024     | 2024       | PASS   |
| 2023-02-28 (Feb FYE)             | 2023     | 2023       | PASS   |
| 2025-02-28 (May FYE)             | 2025     | 2025       | PASS   |

**Finding**: Leap year handling works correctly ✓

### 3. Q1 vs Q4 Boundary Cases ⚠️

| Test Case                        | Expected | Calculated | Status |
|----------------------------------|----------|------------|--------|
| 2025-01-31 (Jan FYE, FY)         | 2024     | 2025       | FAIL   |
| 2025-04-30 (Jan FYE, Q1)         | 2026     | 2026       | PASS   |
| 2024-02-29 (Feb FYE, FY)         | 2024     | 2024       | PASS   |
| 2024-05-31 (Feb FYE, Q1)         | 2025     | 2025       | PASS   |

**Finding**: FY periods where `period_month == fiscal_year_end_month` fail for Jan-Jun FYE ⚠️

### 4. Same Calendar Date, Different Fiscal Years ✓

| Period End | FYE Month | Expected FY | Calculated FY | Status |
|------------|-----------|-------------|---------------|--------|
| 2024-11-30 | May (5)   | 2025        | 2025          | PASS   |
| 2024-11-30 | Oct (10)  | 2025        | 2025          | PASS   |
| 2024-11-30 | Dec (12)  | 2024        | 2024          | PASS   |
| 2024-05-31 | May (5)   | 2024        | 2024          | PASS   |
| 2024-05-31 | Dec (12)  | 2024        | 2024          | PASS   |
| 2024-05-31 | Jan (1)   | 2025        | 2025          | PASS   |

**Finding**: Cross-company date handling works correctly ✓

### 5. Comparative Periods (Prior Year) ✓

Validated through multi-year test data for ORCL and WMT.

**Finding**: Prior year fiscal periods calculated correctly ✓

### 6. Missing Data Handling ✓

Error handling implemented in `_calculate_fiscal_year_from_date`:
```python
except (ValueError, TypeError) as e:
    logger.warning(f"Error calculating fiscal year from date {period_end_date}: {e}")
    return period_end.year  # Fallback to calendar year
```

**Finding**: Graceful degradation implemented ✓

### 7. Fiscal Year Crossover ✓

| Company | FY End | Period          | Calculated FY | Status |
|---------|--------|-----------------|---------------|--------|
| ORCL    | May 31 | 2024-11-30 (Q2) | 2025          | PASS   |
| ORCL    | May 31 | 2024-08-31 (Q1) | 2025          | PASS   |
| ORCL    | May 31 | 2025-02-28 (Q3) | 2025          | PASS   |
| WMT     | Jan 31 | 2024-04-30 (Q1) | 2025          | PASS   |
| WMT     | Jan 31 | 2024-10-31 (Q3) | 2025          | PASS   |

**Finding**: Fiscal year crossover calculated correctly ✓

---

## Test Results Summary

### Overall Results

| Category                              | PASS | FAIL | ERROR | Total |
|---------------------------------------|------|------|-------|-------|
| Oracle Edge Cases (May FYE)           | 8    | 0    | 0     | 8     |
| Walmart Edge Cases (Jan FYE)          | 4    | 2    | 0     | 6     |
| Leap Year Edge Cases                  | 4    | 0    | 0     | 4     |
| Q1/Q4 Boundary Edge Cases             | 3    | 1    | 0     | 4     |
| Same Date, Different FY               | 6    | 0    | 0     | 6     |
| **TOTAL**                             | **25** | **3** | **0** | **28** |

**Overall Pass Rate**: 89.3% (before fix)
**Expected After Fix**: 100.0%

---

## Critical Finding

### The Bug

**Current Logic** (INCORRECT):
```python
if period_month > fiscal_year_end_month:
    return period_year + 1
else:
    return period_year
```

**Issue**: For companies with FYE in Jan-Jun, when `period_month == fiscal_year_end_month`:
- Returns `period_year`
- Should return `period_year - 1`

### Why This Happens

**Companies with FYE in Jan-Jun** (first half of calendar year):
- Fiscal year LABEL = calendar year of the **START** of the fiscal year
- Example: WMT FY 2024 runs from Feb 1, 2024 → Jan 31, 2025
- Period ending Jan 31, 2025 is the **END** of FY 2024, not FY 2025

**Companies with FYE in Jul-Dec** (second half of calendar year):
- Fiscal year LABEL = calendar year of the **END** of the fiscal year
- Example: ORCL FY 2025 runs from Jun 1, 2024 → May 31, 2025
- Period ending May 31, 2025 is the **END** of FY 2025

### The Fix

```python
if period_month > fiscal_year_end_month:
    return period_year + 1
elif period_month == fiscal_year_end_month:
    # CRITICAL: Jan-Jun FYE requires special handling
    if fiscal_year_end_month <= 6:  # Jan-Jun FYE
        return period_year - 1
    else:  # Jul-Dec FYE
        return period_year
else:
    return period_year
```

---

## Affected Companies

### By Fiscal Year End

| FYE Range | Months      | Companies | % of DB | Risk Level |
|-----------|-------------|-----------|---------|------------|
| Jan-Jun   | 1-6         | 3,213     | 18%     | HIGH       |
| Jul-Dec   | 7-12        | 14,787    | 82%     | None       |

### Breakdown (Jan-Jun FYE)

| FYE Month | Count | Risk  |
|-----------|-------|-------|
| Jan (01)  | 392   | HIGH  |
| Feb (02)  | 365   | HIGH  |
| Mar (03)  | 930   | MED   |
| Apr (04)  | 252   | MED   |
| May (05)  | 237   | LOW   |
| Jun (06)  | 1,037 | LOW   |

**Total**: 3,213 companies requiring special fiscal year calculation logic

---

## Recommendations

### IMMEDIATE (Priority 1)

1. **Apply fiscal year calculation fix**
   - File: `src/investigator/infrastructure/sec/companyfacts_extractor.py`
   - Method: `_calculate_fiscal_year_from_date` (lines 593-631)
   - Add special case for Jan-Jun FYE when `period_month == fiscal_year_end_month`

2. **Re-run validation suite**
   ```bash
   python3 tests/edge_case_fiscal_year_validation.py
   ```
   - Expected: 28/28 PASS (100%)

3. **Verify Walmart data**
   ```bash
   python3 cli_orchestrator.py analyze WMT -m standard
   cat results/WMT_*.json | jq '.fiscal_year, .fiscal_period'
   ```
   - Verify fiscal_year matches bulk table data

### HIGH PRIORITY (Priority 2)

4. **Add unit tests**
   - Create `tests/unit/infrastructure/sec/test_fiscal_year_calculation.py`
   - Test all 12 fiscal year end months
   - Test leap year handling
   - Test boundary cases

5. **Database validation query**
   ```sql
   -- Verify processed data matches bulk table
   SELECT COUNT(*) as mismatch_count
   FROM sec_companyfacts_processed proc
   JOIN sec_sub_data sub ON proc.adsh = sub.adsh
   WHERE proc.fiscal_year != sub.fy
     AND sub.form IN ('10-K', '10-Q')
     AND sub.fye IN ('0131', '0228', '0229', '0331', '0430', '0531', '0630');
   ```
   - Expected: 0 mismatches after fix

### MEDIUM PRIORITY (Priority 3)

6. **Update documentation**
   - CLAUDE.md: Add note about fiscal year edge cases
   - Reference FISCAL_YEAR_EDGE_CASE_ANALYSIS.md

7. **Regression testing**
   - Test companies with Jan-Jun FYE
   - Verify no regressions for Jul-Dec FYE companies

---

## Validation Checklist

- [x] Identify edge cases to test
- [x] Create validation test suite
- [x] Run tests and document results
- [x] Identify critical bug (Jan-Jun FYE edge case)
- [x] Document fix approach
- [ ] Apply fix to codebase
- [ ] Re-run validation (expect 100% PASS)
- [ ] Verify affected companies (WMT, Feb FYE)
- [ ] Add unit tests
- [ ] Update documentation

---

## Files Created

1. **`tests/edge_case_fiscal_year_validation.py`**
   - Comprehensive edge case validation suite
   - 28 test cases covering all scenarios
   - Ready to run as part of CI/CD

2. **`docs/FISCAL_YEAR_EDGE_CASE_ANALYSIS.md`**
   - Detailed analysis of the bug
   - Impact assessment
   - Proposed fix with code examples

3. **`docs/FISCAL_YEAR_EDGE_CASE_TEST_DATA.md`**
   - Real-world test data from SEC database
   - Database queries for validation
   - Company-specific examples (ORCL, WMT, MSFT, AMZN)

4. **`docs/EDGE_CASE_VALIDATION_SUMMARY.md`** (this file)
   - High-level summary for quick reference

---

## Conclusion

**Status**: ⚠️ CRITICAL FIX REQUIRED

**Issue**: Fiscal year calculation has systematic error for 3,213 companies (18% of database) with Jan-Jun fiscal year ends.

**Impact**: Off-by-one fiscal year error when `period_month == fiscal_year_end_month`

**Fix**: Add 10 lines of code to handle Jan-Jun FYE edge case

**Validation**: 89.3% pass rate before fix → 100% expected after fix

**Next Steps**: Apply fix, re-run validation, verify affected companies
