# Fiscal Year Initial Assignment - Complete Trace

## Quick Answer

**Where fiscal_year is FIRST set:**
- **File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/infrastructure/sec/data_processor.py`
- **Method**: `process_raw_data()` (lines 1225-1550)
- **Line 1366**: `'fiscal_year': actual_fiscal_year`
- **Value source**: `actual_fiscal_year` derived from `period_end_date.year` at line 1310

**The flow:**
```
CompanyFacts JSON entry['end'] (e.g., "2024-11-30")
  → datetime.strptime() [Line 1305]
  → period_end_date.year [Line 1310]
  → actual_fiscal_year = 2024
  → filings[adsh]['fiscal_year'] = 2024 [Line 1366]
  → Database INSERT [Line ~1544]
```

---

## Root Cause of ORCL Q2-2025 Bug

### The Bug
ORCL Q2 period ending **2024-11-30** is stored with **fiscal_year=2024** (WRONG)
- Should be: **fiscal_year=2025** (ORCL's fiscal year ends May 31)
- Reason: Nov 30 is AFTER May 31, so it's in the next fiscal year

### Why It Happens
Line 1310 uses calendar year of period_end_date:
```python
actual_fiscal_year = period_end_date.year  # 2024 for "2024-11-30"
```

This only works for calendar-year companies (fiscal year = Dec 31).

For non-calendar companies like ORCL (FY ends May 31):
- Nov 30, 2024 is AFTER May 31
- Therefore it belongs to FY2025, not FY2024
- But `period_end_date.year` extracts 2024

### Why the Q1 Fix Doesn't Help
Lines 1344-1361 ONLY apply to Q1 periods:
```python
if actual_fp == 'Q1' and fiscal_year_end:  # ← Q1 ONLY!
    if (period_end_date.month > fy_end_month):
        actual_fiscal_year += 1
```

For ORCL Q2:
- `actual_fp = 'Q2'` (not Q1)
- Condition is FALSE
- No adjustment applied
- Result: fiscal_year remains 2024 (WRONG)

---

## Complete Method Signature

**File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/infrastructure/sec/data_processor.py`

**Method**: `process_raw_data()` at line 1225

```python
def process_raw_data(
    self,
    symbol: str,
    raw_data: Dict,
    raw_data_id: int,
    extraction_version: str = "1.0.0",
    persist: bool = True,
    current_price: Optional[float] = None,
) -> List[Dict]:
    """
    Extract all quarterly/annual filings from raw us-gaap structure

    Args:
        symbol: Stock ticker
        raw_data: Raw SEC API response with us-gaap structure
        raw_data_id: ID from sec_companyfacts_raw table (for lineage)
        extraction_version: Version of extraction logic
        persist: When False, skip writing results to sec_companyfacts_processed
        current_price: Current stock price (optional)

    Returns:
        List of processed filing dicts (one per quarter/year)
    """
```

---

## Key Code Sections

### Section 1: Fiscal Year End Detection (Lines 1286-1291)
```python
# Detect fiscal year end for Q1 fiscal year adjustment
fiscal_year_end = self._detect_fiscal_year_end(raw_data, symbol)
if fiscal_year_end:
    logger.info(f"[Fiscal Year End] {symbol}: Detected fiscal year end: {fiscal_year_end}")
else:
    logger.warning(f"[Fiscal Year End] {symbol}: Could not detect fiscal year end")
```

**What it does**:
- Calls `FiscalPeriodService.detect_fiscal_year_end()` (line 204)
- Returns fiscal year end in format "-MM-DD" (e.g., "-05-31" for ORCL)
- Scans all 10-K entries to find most common FY end date

### Section 2: Period End Date Parsing (Lines 1300-1308)
```python
# Derive actual fiscal year from period_end (not from fy field!)
period_end_str = entry['end']  # e.g., "2024-11-30"
if not period_end_str:
    continue

try:
    period_end_date = datetime.strptime(period_end_str, '%Y-%m-%d')
except ValueError:
    logger.warning(f"Invalid period_end_date format: {period_end_str}")
    continue
```

**What it does**:
- Extracts period_end_str from raw entry (e.g., "2024-11-30")
- Converts to Python datetime object

### Section 3: Initial Fiscal Year Assignment (Line 1310)
```python
actual_fiscal_year = period_end_date.year
```

**This is the EXACT LINE where fiscal_year is first set!**

**For ORCL Q2-2025**:
- Input: `period_end_date = datetime(2024, 11, 30)`
- Result: `actual_fiscal_year = 2024`
- Problem: Nov 30, 2024 is actually FY2025 for ORCL (FY ends May 31)

### Section 4: Fiscal Period Derivation (Lines 1312-1336)
```python
# Derive fiscal period using fp field (authoritative after filtering)
duration = entry.get('duration_days', 999)
raw_fp = entry.get('fp', '')

# Use fp from entry if available and valid
if raw_fp == 'FY' or duration >= 330:
    actual_fp = 'FY'
elif raw_fp in ['Q1', 'Q2', 'Q3', 'Q4']:
    actual_fp = raw_fp
else:
    # Fallback: derive quarter from end month
    month = period_end_date.month
    if month <= 3:
        actual_fp = 'Q1'
    elif month <= 6:
        actual_fp = 'Q2'
    elif month <= 9:
        actual_fp = 'Q3'
    else:
        actual_fp = 'Q4'
```

**For ORCL Q2-2025**:
- `raw_fp = 'Q2'` (from CompanyFacts API)
- `duration >= 330`? No (Q2 is ~91 days)
- Result: `actual_fp = 'Q2'`

### Section 5: Q1-Only Fiscal Year Adjustment (Lines 1338-1361)
```python
# CRITICAL FIX: Adjust fiscal_year for Q1 periods in non-calendar fiscal years
# Q1 can cross calendar year boundary. If period_end is after fiscal_year_end,
# Q1 belongs to the NEXT fiscal year.
if actual_fp == 'Q1' and fiscal_year_end:
    try:
        # Extract month and day from fiscal_year_end (format: '-MM-DD')
        fy_end_month, fy_end_day = map(int, fiscal_year_end[1:].split('-'))
        
        # Check if period_end is after fiscal_year_end
        # If so, Q1 belongs to the next fiscal year
        if (period_end_date.month > fy_end_month) or \
           (period_end_date.month == fy_end_month and period_end_date.day > fy_end_day):
            original_fy = actual_fiscal_year
            actual_fiscal_year += 1
            logger.debug(
                f"[Q1 Fiscal Year Adjustment] {symbol} Q1 ending {period_end_str}: "
                f"Adjusted fiscal_year from {original_fy} to {actual_fiscal_year} "
                f"(fiscal year ends {fiscal_year_end})"
            )
    except Exception as e:
        logger.warning(f"[Q1 Fiscal Year Adjustment] {symbol}: Failed to adjust Q1 fiscal year: {e}")
```

**For ORCL Q2-2025**:
- Condition: `if actual_fp == 'Q1' and fiscal_year_end`
- `actual_fp = 'Q2'` (not Q1)
- Condition evaluates to FALSE
- NO adjustment applied
- BUG: fiscal_year remains 2024

### Section 6: Filing Dictionary Creation (Lines 1363-1378)
```python
filings[adsh] = {
    'symbol': symbol.upper(),
    'cik': cik,
    'fiscal_year': actual_fiscal_year,  # ← LINE 1366: ASSIGNED HERE
    'fiscal_period': actual_fp,
    'adsh': adsh,
    'form_type': entry['form'],
    'filed_date': entry['filed'],
    'period_end_date': period_end_str,
    'period_start_date': entry['start'],
    'frame': entry['frame'],
    'duration_days': duration,
    'data': {},
    'raw_data_id': raw_data_id,
    'extraction_version': extraction_version
}
```

**This is where the fiscal_year value becomes persistent in the data structure.**

---

## Where It Gets Used

### 1. Data Extraction (Line 1444)
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

### 2. Database Write (Line ~1544)
```python
if persist:
    self._persist_processed_filings(processed_filings, symbol, raw_data_id, extraction_version)
```

The `_persist_processed_filings()` method writes the fiscal_year to the database:
```sql
INSERT INTO sec_companyfacts_processed (
    symbol, cik, fiscal_year, fiscal_period, adsh, ...
) VALUES (
    'ORCL', '0001633917', 2024, 'Q2', '0001632280-24-035973', ...
)  -- fiscal_year=2024 (INCORRECT!)
```

---

## Fiscal Year End Detection Details

**File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/domain/services/fiscal_period_service.py`

**Method**: `detect_fiscal_year_end()` (lines 230-301)

**Algorithm**:
1. Scan all concepts in us-gaap, dei, ifrs-full taxonomies
2. Find all entries where `form='10-K'` and `fy` field exists
3. Extract period_end dates (format: "YYYY-MM-DD")
4. Extract month-day suffix (e.g., "2024-05-31" → "-05-31")
5. Count occurrences of each suffix
6. Return most common suffix

**For ORCL**:
1. Finds 10-K entries ending: 2025-05-31, 2024-05-31, 2023-05-31, ...
2. All have suffix: "-05-31"
3. Returns: "-05-31"

---

## Data Input: Where Raw Entries Come From

The raw entries with 'fy', 'fp', 'end', etc. come from the SEC CompanyFacts API:

```json
{
  "facts": {
    "us-gaap": {
      "Assets": {
        "units": {
          "USD": [
            {
              "val": 1234567890,
              "end": "2024-11-30",           # ← Period end date
              "filed": "2024-12-20",
              "form": "10-Q",               # ← Form type
              "fy": 2024,                   # ← SEC's fiscal year (UNRELIABLE)
              "fp": "Q2",                   # ← SEC's fiscal period (UNRELIABLE)
              "frame": "CY2024Q3",
              "accn": "0001632280-24-035973",
              "start": "2024-06-01"         # ← Period start date
            }
          ]
        }
      }
    }
  }
}
```

These entries flow through:
1. `companyfacts_extractor.py` - Fetches and caches raw data
2. `data_processor.py._discover_all_period_entries()` - Discovers unique periods
3. `data_processor.py._select_best_entries_per_period()` - Filters and selects best
4. `data_processor.py.process_raw_data()` - Processes and assigns fiscal_year

---

## Summary Table

| Component | File | Line(s) | Purpose |
|-----------|------|---------|---------|
| **Initial Assignment** | data_processor.py | 1310 | Sets fiscal_year = period_end_date.year |
| **Q1 Adjustment Logic** | data_processor.py | 1344-1361 | Adjusts fiscal_year for Q1 only (BUG HERE!) |
| **Filing Dictionary** | data_processor.py | 1366 | Stores fiscal_year in filing dict |
| **Data Extraction** | data_processor.py | 1444 | Uses fiscal_year to extract metrics |
| **Database Write** | data_processor.py | ~1544 | Persists fiscal_year to database |
| **Fiscal Year End Detection** | fiscal_period_service.py | 230-301 | Detects FY end from 10-K entries |
| **Detection Wrapper** | data_processor.py | 204-224 | Calls FiscalPeriodService.detect() |

---

## The Fix Required

**Generalize the Q1-only adjustment to ALL quarters** at lines 1344-1361:

```python
# CURRENT (Q1 ONLY - BUGGY)
if actual_fp == 'Q1' and fiscal_year_end:

# PROPOSED (ALL QUARTERS - FIX)
if fiscal_year_end and actual_fp in ['Q1', 'Q2', 'Q3', 'Q4']:
```

This ensures that for ALL periods in non-calendar fiscal year companies, if the period_end is after the fiscal_year_end date, the fiscal_year is incremented.

---

## Related Documentation Files

Created in `/Users/vijaysingh/code/InvestiGator/docs/`:

1. **FISCAL_YEAR_ASSIGNMENT_TRACE.md** - High-level trace with bug explanation
2. **FISCAL_YEAR_CODE_LOCATIONS.md** - Detailed code snippets with line numbers

