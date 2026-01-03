# SEC Data Processing Bug Analysis

**Date**: 2025-11-09
**Issue**: All quarters for a fiscal year have same period_end_date in processed table
**Impact**: NULL financial data, failed quarterly calculations, incorrect valuations

---

## Bug Symptoms

### AAPL Processed Table (WRONG)
```
fiscal_year | fiscal_period | period_end_date | ocf_millions
------------|---------------|-----------------|-------------
2025        | Q1            | 2024-09-28      | 39,895
2025        | Q2            | 2024-09-28      | 22,690
2025        | Q3            | 2024-09-28      | 28,858
2025        | FY            | 2024-09-28      | 118,254
```

**All have 2024-09-28 (FY end date)!**

### AAPL Raw Data (CORRECT)
```
fy   | fp | period_end  | ocf_millions
-----|----|-----------|--------------
2025 | Q1 | 2024-12-28  | 29,935 ✓
2025 | Q2 | 2025-03-29  | 53,887 ✓ (YTD)
2025 | Q3 | 2025-06-28  | 81,754 ✓ (YTD)
2025 | FY | 2024-09-28  | 118,254 ✓
```

**Each period has correct end date!**

---

## Root Cause

**File**: `src/investigator/infrastructure/sec/data_processor.py`
**Location**: Lines 708-748 (PHASE 1: Discover filings)

### Problematic Code

```python
# PHASE 1: Discover all filings (ADSH) in the dataset
for tag_name, tag_data in us_gaap.items():
    units = tag_data.get('units', {})
    usd_data = units.get('USD', [])

    for entry in usd_data:
        form = entry.get('form', '')
        if form not in ['10-K', '10-Q']:
            continue

        adsh = entry.get('accn', '')
        if not adsh or adsh in filings:  # ❌ SKIP if ADSH exists
            continue

        # Initialize filing entry
        filings[adsh] = {
            'period_end_date': entry.get('end'),  # ❌ Uses FIRST tag's end date
            ...
        }
```

### Why It's Wrong

1. **First tag wins**: Uses `period_end_date` from whichever tag we scan first
2. **All tags skipped after first**: `if adsh in filings: continue` prevents updating
3. **FY contaminates quarters**: If FY filing's tag is scanned first, all quarters inherit FY's end date

**Example**:
- Scanning `TotalRevenue` tag first
- Finds FY entry: adsh=123, end=2024-09-28
- Creates filing: `filings['123'] = {period_end_date: '2024-09-28'}`
- Later finds Q1 entry: adsh=123, end=2024-12-28
- **SKIPPED** because `adsh in filings`
- Q1 keeps wrong period_end_date=2024-09-28

---

## Additional Issues Found

### Issue 1: Multiple Filings for Same Period (Restatements)

**AAPL Raw**:
```
fy   | fp | filed      | period_end  | ocf_millions
-----|----|-----------|-----------|--------------
2025 | FY | 2025-10-31 | 2024-09-28  | 118,254  (latest)
2024 | FY | 2024-11-01 | 2024-09-28  | 118,254  (older)
```

**Same period_end, different SEC fy labels, different filed dates!**

**Current behavior**: Keeps first encountered (random)
**Should**: Keep LATEST filed date

### Issue 2: Multiple Values for Same Period (YTD vs Individual)

**MSFT Raw** (Q3 2025-03-31):
```
fy   | fp | filed      | period_end  | ocf_millions
-----|----|-----------|-----------|--------------
2025 | Q3 | 2025-04-30 | 2025-03-31  | 93,515  (YTD cumulative)
2025 | Q3 | 2025-04-30 | 2025-03-31  | 37,044  (individual quarter)
```

**Same filed date, same period, different values!**

**Current behavior**: Uses first value encountered (random)
**Should**: Select based on duration (prefer individual quarter < 120 days)

### Issue 3: SEC's fy Field is Unreliable

**STX Example**:
```
period_end  | SEC fy label | Actual fiscal year
------------|--------------|-------------------
2024-06-28  | fy=2027      | 2024 (June FY)
2025-06-27  | fy=2026      | 2025 (June FY)
```

**SEC labels fiscal year WRONG for non-calendar fiscal years!**

---

## Proposed Fix

### Strategy 1: Proper Period End Date Selection

Instead of using first tag's `end` date, select period_end_date by:

1. **Scan ALL entries for ADSH** (don't skip after first)
2. **Filter to matching fiscal period** (fp='Q1' entries only for Q1 filing)
3. **Prefer shortest duration** (<120 days = individual quarter, not YTD)
4. **Use most common end date** if multiple values exist

**Pseudocode**:
```python
# PHASE 1: Discover all filings
for tag_name, tag_data in us_gaap.items():
    for entry in usd_data:
        adsh = entry.get('accn', '')
        fp = entry.get('fp')

        # Collect ALL entries for this ADSH
        if adsh not in filings:
            filings[adsh] = {
                'adsh': adsh,
                'fp': fp,
                'period_end_candidates': []
            }

        # Add period_end candidate with duration
        start = entry.get('start')
        end = entry.get('end')
        if start and end:
            days = (parse_date(end) - parse_date(start)).days
            filings[adsh]['period_end_candidates'].append({
                'end': end,
                'duration_days': days,
                'fp': entry.get('fp')
            })

# PHASE 1.5: Select best period_end_date for each filing
for adsh, filing in filings.items():
    candidates = filing['period_end_candidates']

    # Filter to matching fiscal period
    fp_matches = [c for c in candidates if c['fp'] == filing['fp']]

    # Prefer shortest duration (individual quarter)
    fp_matches.sort(key=lambda c: c['duration_days'])

    if fp_matches:
        filing['period_end_date'] = fp_matches[0]['end']
    else:
        filing['period_end_date'] = candidates[0]['end']  # Fallback
```

### Strategy 2: Deduplication by (period_end, fp, form)

Handle restatements by keeping LATEST filing:

```python
# After collecting all filings, deduplicate
unique_filings = {}
for adsh, filing in filings.items():
    key = (filing['period_end_date'], filing['fp'], filing['form_type'])

    if key not in unique_filings:
        unique_filings[key] = filing
    else:
        # Keep filing with LATEST filed_date
        if filing['filed_date'] > unique_filings[key]['filed_date']:
            unique_filings[key] = filing

filings = unique_filings
```

### Strategy 3: Derive fiscal_year from period_end_date

Already implemented in current code! Just needs to work with correct period_end_date.

---

## Testing Plan

### Test 1: AAPL Period End Dates

**Expected after fix**:
```
fiscal_year | fiscal_period | period_end_date | ocf_millions
------------|---------------|-----------------|-------------
2025        | Q1            | 2024-12-28      | 29,935
2025        | Q2            | 2025-03-29      | 23,952 (Q2 individual)
2025        | Q3            | 2025-06-28      | 27,867 (Q3 individual)
2025        | FY            | 2024-09-28      | 118,254
```

### Test 2: STX Fiscal Year Correction

**Expected after fix**:
```
fiscal_year | fiscal_period | period_end_date | SEC fy label
------------|---------------|-----------------|-------------
2024        | FY            | 2024-06-28      | 2027 (corrected)
2025        | Q1            | 2025-09-27      | 2026 (corrected)
```

### Test 3: Q4 Computation

With correct period_end_dates, Q4 computation should work:
```
Q4-2024 = FY-2024 - (Q1-2024 + Q2-2024 + Q3-2024)
```

---

## Priority

**CRITICAL** - This bug affects:
- All quarterly calculations
- All TTM metrics
- All DCF valuations
- All companies (100% impact)

**Recommended Action**: Fix immediately, re-process all symbols

---

## Files to Modify

1. `src/investigator/infrastructure/sec/data_processor.py` - PHASE 1 filing discovery
2. `utils/quarterly_calculator.py` - Verify works correctly after fix
3. Test with: AAPL, MSFT, STX (calendar and non-calendar fiscal years)
