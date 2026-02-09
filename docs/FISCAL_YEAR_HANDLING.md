# Fiscal Year Handling - Comprehensive Guide

## Overview

This document provides comprehensive guidance on fiscal year handling in the InvestiGator system, including the data flow, known issues, and implementation details.

## Quick Reference

**Where fiscal_year is initially set**: `data_processor.py` line 1366 (filing dictionary creation), derived from line 1310 (`actual_fiscal_year = period_end_date.year`)

**The Bug**: For non-calendar fiscal year companies (e.g., ORCL with May 31 FY end), Q2-Q4 periods get assigned the wrong fiscal year because the code assumes calendar years.

## Table of Contents

1. [Data Flow](#data-flow)
2. [Code Locations](#code-locations)
3. [The Q1-Only Bug](#the-q1-only-bug)
4. [Edge Cases](#edge-cases)
5. [Implementation Guide](#implementation-guide)

---

## Data Flow

```
CompanyFacts API entry['end'] = "2024-11-30"
    ↓
process_raw_data() [Line 1305]
    period_end_date = datetime.strptime(period_end_str, '%Y-%m-%d')
    → datetime(2024, 11, 30)
    ↓
process_raw_data() [Line 1310] ← INITIAL ASSIGNMENT
    actual_fiscal_year = period_end_date.year = 2024
    ↓
process_raw_data() [Line 1287]
    fiscal_year_end = _detect_fiscal_year_end(raw_data, symbol)
    → "-05-31" for ORCL
    ↓
process_raw_data() [Line 1344]
    if actual_fp == 'Q1' and fiscal_year_end:  # Q1-ONLY BUG!
        # Adjustment logic
    ↓
process_raw_data() [Line 1366] ← STORED IN DICT
    filings[adsh]['fiscal_year'] = actual_fiscal_year
    ↓
Database [Line ~1544]
    INSERT INTO sec_companyfacts_processed (fiscal_year=...)
```

---

## Code Locations

### 1. Initial Fiscal Year Assignment

**File**: `src/investigator/infrastructure/sec/data_processor.py`
**Method**: `process_raw_data()`
**Lines**: 1225-1550 (full method)

**Key Line 1310** (THE PROBLEM):
```python
actual_fiscal_year = period_end_date.year
```
This assumes calendar-year fiscal years. For ORCL (FY ends May 31):
- Nov 30, 2024 is AFTER May 31
- Therefore it belongs to FY2025, not FY2024
- But `period_end_date.year` extracts 2024

### 2. Fiscal Year End Detection

**File**: `src/investigator/infrastructure/sec/data_processor.py`
**Method**: `_detect_fiscal_year_end()`
**Lines**: 204-224

Delegates to `FiscalPeriodService.detect_fiscal_year_end()` which:
- Finds FY (10-K) filings in company facts
- Extracts period_end dates
- Determines most common month-day suffix (e.g., "-05-31" for ORCL)

**Service**: `src/investigator/domain/services/fiscal_period_service.py`
**Method**: `detect_fiscal_year_end()`
**Lines**: 230-301

### 3. Q1-Only Fiscal Year Adjustment

**File**: `src/investigator/infrastructure/sec/data_processor.py`
**Lines**: 1344-1361

```python
if actual_fp == 'Q1' and fiscal_year_end:
    # Detect fiscal year end month/day
    fy_end_month, fy_end_day = map(int, fiscal_year_end[1:].split('-'))

    # Check if period_end is after fiscal_year_end
    if (period_end_date.month > fy_end_month) or \
       (period_end_date.month == fy_end_month and period_end_date.day > fy_end_day):
        actual_fiscal_year += 1
```

**THE BUG**: This only applies to Q1 periods! Q2, Q3, Q4 are not adjusted.

### 4. Filing Dictionary Creation

**File**: `src/investigator/infrastructure/sec/data_processor.py`
**Line**: 1366

```python
filings[adsh] = {
    'symbol': symbol.upper(),
    'cik': cik,
    'fiscal_year': actual_fiscal_year,  # ← STORED HERE
    'fiscal_period': actual_fp,
    # ... other fields
}
```

### 5. Data Extraction

**File**: `src/investigator/infrastructure/sec/data_processor.py`
**Lines**: 1444

```python
value, source_tag = self._extract_from_json_for_filing(
    canonical_key,
    us_gaap,
    adsh,
    fiscal_year=filing['fiscal_year'],  # ← USED HERE
    fiscal_period=filing['fiscal_period'],
    period_end=filing.get('period_end_date'),
)
```

---

## The Q1-Only Bug

### Problem

For companies with non-calendar fiscal years (e.g., ORCL ends May 31), Q2-Q4 periods get the wrong fiscal_year.

### Example: ORCL Q2

```
Period: Q2 ending 2024-11-30
Fiscal Year End: May 31 (-05-31)

Expected: fiscal_year = 2025 (Nov 30 > May 31)
Actual:   fiscal_year = 2024 (period_end_date.year = 2024)

Why Q1 fix doesn't help:
- Line 1344: if actual_fp == 'Q1' and fiscal_year_end:
- For Q2: actual_fp = 'Q2', condition is FALSE
- No adjustment applied → fiscal_year stays wrong
```

### Impact

- YTD grouping failures
- Consecutive quarter validation breaks (365-day gaps)
- DCF valuations use wrong data
- Growth rate calculations span wrong periods

### The Fix

Change lines 1344-1361 from Q1-only to ALL quarters:

```python
# BEFORE (BUGGY)
if actual_fp == 'Q1' and fiscal_year_end:

# AFTER (FIXED)
if fiscal_year_end and actual_fp in ['Q1', 'Q2', 'Q3', 'Q4']:
```

This ensures ALL quarters in non-calendar FY companies get adjusted when `period_end` is after `fiscal_year_end`.

---

## Edge Cases

### Q1 Periods Crossing Calendar Year Boundary

**Example**: ZS (FY ends July 31)

| Period End | FY End | Before Fix | After Fix | Correct? |
|------------|--------|------------|-----------|----------|
| 2023-10-31 | -07-31 | 2023 | 2024 | ✅ After fix |
| 2022-10-31 | -07-31 | 2022 | 2023 | ✅ After fix |

**Rule**: If `period_end_date` > `fiscal_year_end`, increment `fiscal_year`.

### Calendar Year Companies

For companies with December 31 FY end (e.g., AAPL), the adjustment logic works correctly but is unnecessary since `period_end_date.year` already equals the correct fiscal year.

### Edge Case: Same Day as FY End

If `period_end_date` equals `fiscal_year_end` exactly, no adjustment needed (the period belongs to the current fiscal year).

---

## Implementation Guide

### How to Verify the Bug

```sql
-- Check for wrong fiscal_year assignments
SELECT symbol, fiscal_year, fiscal_period, period_end_date
FROM sec_companyfacts_processed
WHERE symbol = 'ORCL'
  AND fiscal_period IN ('Q2', 'Q3', 'Q4')
ORDER BY period_end_date DESC;

-- Expected for ORCL Q2 ending Nov 30, 2024:
-- fiscal_year should be 2025 (because Nov 30 > May 31)
```

### Testing Checklist

- [ ] Verify Q1 fiscal_year adjustment works
- [ ] Verify Q2-Q4 fiscal_year adjustment works
- [ ] Test with multiple non-calendar FY companies
- [ ] Verify YTD grouping succeeds with corrected fiscal_year
- [ ] Verify consecutive quarter validation passes
- [ ] Run DCF valuation with corrected data

### Related Code

- `src/investigator/infrastructure/sec/data_processor.py:1250-1325` (Q1 fix for CompanyFacts path)
- `utils/sec_data_strategy.py:461-507, 568-610` (Q1 fix for bulk tables path)
- `src/investigator/domain/services/fiscal_period_service.py:230-301` (FY end detection)

---

## Related Documentation

- `CLI_DATA_COMMANDS.md` - CLI commands for cache management
- `ARCHITECTURE_DECISION_DATA_ACCESS.md` - Data access layer decisions
- `VALUATION_CONFIGURATION.md` - Valuation config reference

---

**Last Updated**: 2025-11-13
**Status**: Bug identified and documented. Fix applied to both CompanyFacts and bulk table paths.
