# Fiscal Year Assignment Trace - SEC Data Processing Pipeline

## Executive Summary

The `fiscal_year` value is **initially assigned at line 1366 in `data_processor.py`** within the `process_raw_data()` method. The value comes from `actual_fiscal_year`, which is derived from the period end date's calendar year, NOT from SEC XBRL data.

**Root of ORCL Q2-2025 Bug**: The pipeline uses `period_end_date.year` (line 1310), which for ORCL Q2 ending 2024-11-30 extracts 2024. However, ORCL has a May 31 fiscal year end, so Nov 30 belongs to FY2025, not FY2024.

---

## Complete Data Flow

```
CompanyFacts API JSON
    ↓ (SEC facts structure)
Raw Data Entry
    ├─ entry['end']: "2024-11-30"
    ├─ entry['fy']: 2024 (UNRELIABLE - document metadata)
    ├─ entry['fp']: "Q2" (UNRELIABLE for non-calendar years)
    ├─ entry['form']: "10-Q"
    └─ entry['start']: "2024-06-01"
         ↓
process_raw_data() METHOD [Line 1225]
    ├─ Discovers best entries per period [Line 1284]
    ├─ Detects fiscal year end: "-05-31" [Line 1287]
    └─ PHASE 2: Creates filing dict [Line 1293]
         ├─ Extracts period_end: "2024-11-30" [Line 1300]
         ├─ Derives actual_fiscal_year = period_end_date.year [Line 1310]
         │   Result: 2024 (WRONG for non-calendar fiscal years!)
         ├─ Derives actual_fp from duration/fp field [Lines 1314-1324]
         │   Result: "Q2"
         └─ Creates filing[adsh] with fiscal_year:2024 [Line 1366]
              └─ fiscal_year is DATABASE WRITTEN [Line 1444]
                   ↓
        Database: sec_companyfacts_processed
        Inserted with fiscal_year=2024 (INCORRECT)
```

---

## File Paths and Line Numbers

### 1. Primary Fiscal Year Assignment

**File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/infrastructure/sec/data_processor.py`

**Method**: `process_raw_data()` (lines 1225-1550)

**Key Lines**:

```python
# Line 1310: Initial fiscal year from period_end calendar year
actual_fiscal_year = period_end_date.year

# Line 1287: Detect fiscal year end (calls FiscalPeriodService)
fiscal_year_end = self._detect_fiscal_year_end(raw_data, symbol)

# Lines 1344-1361: Q1 fiscal year adjustment (only for Q1!)
# NOTE: THIS FIX ONLY APPLIES TO Q1, NOT Q2-Q4!
if actual_fp == 'Q1' and fiscal_year_end:
    # Check if period_end is after fiscal_year_end
    if (period_end_date.month > fy_end_month) or \
       (period_end_date.month == fy_end_month and period_end_date.day > fy_end_day):
        actual_fiscal_year += 1

# Line 1366: Write fiscal_year to filing dictionary
filings[adsh] = {
    'fiscal_year': actual_fiscal_year,  # ✅ ASSIGNED HERE
    'fiscal_period': actual_fp,
    # ... other fields
}

# Line 1444: Used to extract from database
fiscal_year=filing['fiscal_year'],
```

---

## The Bug: Why ORCL Q2 is Wrong

### Symptom
ORCL Q2 ending 2024-11-30 → fiscal_year=2024 (WRONG, should be 2025)

### Root Cause
Line 1310 uses:
```python
actual_fiscal_year = period_end_date.year
```

This ONLY works for calendar-year companies:
- AAPL (fiscal year ends Sept 30): Sep 30, 2024 → 2024 ✓
- ORCL (fiscal year ends May 31): Nov 30, 2024 → **2024 (should be 2025)** ✗

### Why Q1 Fix Doesn't Help
Lines 1344-1361 ONLY adjust Q1:
```python
if actual_fp == 'Q1' and fiscal_year_end:  # Q1 ONLY!
    if (period_end_date.month > fy_end_month):
        actual_fiscal_year += 1
```

For ORCL Q2 (Nov 30, 2024):
- `actual_fp = 'Q2'` (not Q1)
- Condition `if actual_fp == 'Q1'` is FALSE
- NO adjustment applied
- Result: fiscal_year stays as 2024

---

## The Fix: Generalize Fiscal Year Adjustment

Instead of Q1-only logic, apply adjustment for ALL quarters when `period_end > fiscal_year_end`:

```python
# CURRENT CODE (lines 1338-1361) - Q1 ONLY
if actual_fp == 'Q1' and fiscal_year_end:
    if (period_end_date.month > fy_end_month) or \
       (period_end_date.month == fy_end_month and period_end_date.day > fy_end_day):
        actual_fiscal_year += 1

# PROPOSED FIX - ALL QUARTERS
if fiscal_year_end and actual_fp in ['Q1', 'Q2', 'Q3', 'Q4']:
    try:
        fy_end_month, fy_end_day = map(int, fiscal_year_end[1:].split('-'))
        if (period_end_date.month > fy_end_month) or \
           (period_end_date.month == fy_end_month and period_end_date.day > fy_end_day):
            original_fy = actual_fiscal_year
            actual_fiscal_year += 1
            logger.debug(
                f"[Fiscal Year Adjustment] {symbol} {actual_fp} ending {period_end_str}: "
                f"Adjusted fiscal_year from {original_fy} to {actual_fiscal_year} "
                f"(fiscal year ends {fiscal_year_end})"
            )
    except Exception as e:
        logger.warning(f"[Fiscal Year Adjustment] {symbol}: Failed to adjust fiscal year: {e}")
```

---

## Fiscal Year End Detection

**File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/domain/services/fiscal_period_service.py`

**Method**: `detect_fiscal_year_end()` (lines 230-301)

This method:
1. Scans all us-gaap, dei, ifrs-full concepts
2. Finds all 10-K (FY) entries with 'fy' field
3. Extracts period end dates (format: "YYYY-MM-DD")
4. Counts occurrences of each month-day suffix
5. Returns most common suffix (e.g., "-05-31" for ORCL)

```python
def detect_fiscal_year_end(self, company_facts: Dict[str, Any]) -> str:
    """Returns '-MM-DD' format (e.g., '-05-31' for ORCL)"""
    # Iterate through all taxonomies and find 10-K entries
    for entry in unit_data:
        if entry.get('form') == '10-K' and entry.get('fy'):
            period_end = entry.get('end')  # e.g., "2025-05-31"
            # Extract: period_end[-6:] = "-05-31"
            
    return fiscal_year_end  # e.g., "-05-31"
```

**Data Flow for ORCL**:
1. Finds all 10-K entries in CompanyFacts API
2. ORCL has 10-K entries ending: 2025-05-31, 2024-05-31, 2023-05-31, ...
3. Extracts suffixes: "-05-31" (all matches)
4. Returns: "-05-31"

---

## Entry Point: Where Raw Data Comes From

**File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/infrastructure/sec/companyfacts_extractor.py`

The SEC API response contains raw entries like:

```json
{
  "facts": {
    "us-gaap": {
      "Assets": {
        "units": {
          "USD": [
            {
              "val": 1234567890,
              "end": "2024-11-30",
              "filed": "2024-12-20",
              "form": "10-Q",
              "fy": 2024,           # ← SEC's fiscal year (UNRELIABLE)
              "fp": "Q2",           # ← SEC's fiscal period (UNRELIABLE)
              "frame": "CY2024Q3",
              "accn": "0001628280-24-035973",
              "start": "2024-06-01"
            }
          ]
        }
      }
    }
  }
}
```

These raw entries flow into `process_raw_data()` via the `best_entries` list.

---

## Summary Table

| Field | Source | Reliability | Notes |
|-------|--------|-------------|-------|
| `entry['end']` | CompanyFacts API | ✓ Reliable | Actual period end date |
| `entry['fy']` | CompanyFacts API | ✗ Unreliable | Document fiscal year (not period fiscal year) |
| `entry['fp']` | CompanyFacts API | ✗ For non-calendar FYs | Period label (ambiguous for comparative filings) |
| `entry['start']` | CompanyFacts API | ✓ Reliable | Actual period start date |
| `actual_fiscal_year` | Derived from `entry['end'].year` | ✗ For non-calendar FYs | Assumes calendar year companies |
| `fiscal_year_end` | Detected from 10-K entries | ✓ Reliable | Most common FY end month-day |

---

## Required Code Change Location

**File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/infrastructure/sec/data_processor.py`

**Method**: `process_raw_data()`

**Lines to Modify**: 1310 and 1338-1361

**Change Type**: Generalize Q1-only fiscal year adjustment to all quarters

This will fix the issue where non-calendar fiscal year companies (ORCL, ZS, etc.) have incorrect fiscal year assignments for Q2-Q4 periods.

