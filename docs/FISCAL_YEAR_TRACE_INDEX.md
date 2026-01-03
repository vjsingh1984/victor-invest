# Fiscal Year Assignment Trace - Complete Documentation Index

## Overview

This directory contains comprehensive documentation tracing where `fiscal_year` is initially set in the SEC data processing pipeline, including the root cause analysis of the ORCL Q2-2025 bug.

**Quick Answer**: `fiscal_year` is first assigned at **line 1366** in `data_processor.py` within the `process_raw_data()` method, derived from `period_end_date.year` (line 1310).

---

## Documentation Files

### 1. FISCAL_YEAR_QUICK_REFERENCE.txt (START HERE!)
**Best for**: Quick lookup, understanding the bug at a glance

Contains:
- One-page reference showing where fiscal_year is set
- The bug symptom and root cause
- Complete data flow diagram
- Key file locations and line numbers
- The exact fix required

**Read time**: 2-3 minutes

---

### 2. FISCAL_YEAR_INITIAL_ASSIGNMENT_SUMMARY.md (COMPREHENSIVE)
**Best for**: Detailed understanding, implementation guide

Contains:
- Quick answer section
- Root cause analysis with examples
- Complete method signature
- 6 detailed code sections with explanations:
  1. Fiscal Year End Detection (lines 1286-1291)
  2. Period End Date Parsing (lines 1300-1308)
  3. **Initial Fiscal Year Assignment (line 1310)** - THE KEY LINE
  4. Fiscal Period Derivation (lines 1312-1336)
  5. Q1-Only Fiscal Year Adjustment (lines 1344-1361) - THE BUG
  6. Filing Dictionary Creation (lines 1363-1378)
- Where it gets used (extraction, database writes)
- Fiscal year end detection algorithm
- Data input source (SEC CompanyFacts API)
- Summary table of all components
- The fix required

**Read time**: 10-15 minutes

---

### 3. FISCAL_YEAR_ASSIGNMENT_TRACE.md (HIGH-LEVEL)
**Best for**: Understanding data flow, architectural overview

Contains:
- Executive summary
- Visual data flow diagram
- File paths and line numbers
- The bug explanation (ORCL Q2 case study)
- Why the Q1 fix doesn't help
- Proposed fix with code comparison
- Fiscal year end detection method
- Entry point (where raw data comes from)
- Summary table with reliability ratings

**Read time**: 8-10 minutes

---

### 4. FISCAL_YEAR_CODE_LOCATIONS.md (DETAILED CODE)
**Best for**: Code review, implementation, copy-paste reference

Contains:
- 5 detailed code locations with full context:
  1. Primary Fiscal Year Assignment (data_processor.py:1225-1550)
     - Lines 1310, 1344-1361, 1366
  2. Where fiscal_year is Used to Extract Data (lines 1424-1451)
  3. Where fiscal_year is Written to Database (lines ~1500-1550)
  4. Fiscal Year End Detection Service (fiscal_period_service.py:230-301)
  5. Where fiscal_year_end is Retrieved (data_processor.py:204-224)
- Complete method signatures
- Full code snippets with line numbers
- Summary data flow diagram

**Read time**: 12-15 minutes

---

## Key Locations Summary

| What | File | Line(s) | Code |
|------|------|---------|------|
| **Initial Assignment** | `data_processor.py` | 1310 | `actual_fiscal_year = period_end_date.year` |
| **Filing Dict** | `data_processor.py` | 1366 | `'fiscal_year': actual_fiscal_year` |
| **Q1 Adjustment** | `data_processor.py` | 1344-1361 | `if actual_fp == 'Q1' and fiscal_year_end:` |
| **FY End Detection** | `fiscal_period_service.py` | 230-301 | `detect_fiscal_year_end()` |
| **Method Start** | `data_processor.py` | 1225 | `def process_raw_data()` |

---

## The Bug at a Glance

**Symptom**: ORCL Q2 ending 2024-11-30 → fiscal_year=2024 (WRONG - should be 2025)

**Root Cause**: Line 1310 uses `period_end_date.year` which extracts calendar year
- Assumes all companies have calendar fiscal year (Dec 31)
- ORCL has May 31 fiscal year end
- Nov 30, 2024 is AFTER May 31, so it's FY2025, not FY2024

**Why Q1 Fix Doesn't Help**: Lines 1344-1361 only adjust Q1 periods
- Condition: `if actual_fp == 'Q1' and fiscal_year_end:` is Q1-only
- For Q2: condition is FALSE, no adjustment applied

**The Fix**: Change condition from Q1-only to ALL quarters:
```python
# CURRENT (BUGGY)
if actual_fp == 'Q1' and fiscal_year_end:

# PROPOSED (FIX)
if fiscal_year_end and actual_fp in ['Q1', 'Q2', 'Q3', 'Q4']:
```

---

## Data Flow Summary

```
CompanyFacts API entry['end'] = "2024-11-30"
    ↓
process_raw_data() [Line 1305]
datetime.strptime() → datetime(2024, 11, 30)
    ↓
process_raw_data() [Line 1310] ← INITIAL ASSIGNMENT
actual_fiscal_year = period_end_date.year = 2024
    ↓
process_raw_data() [Line 1344]
if actual_fp == 'Q1': [Q2 SKIPS - BUG!]
    ↓
process_raw_data() [Line 1366]
filings[adsh]['fiscal_year'] = 2024 (INCORRECT)
    ↓
Database [Line ~1544]
INSERT INTO sec_companyfacts_processed (fiscal_year=2024)
    ↓
Result: ORCL Q2 stored with fiscal_year=2024 (WRONG!)
```

---

## Files Involved

### Primary Files
1. **`src/investigator/infrastructure/sec/data_processor.py`** (2500+ lines)
   - `process_raw_data()` method: lines 1225-1550
   - Initial assignment: line 1310
   - Filing dict creation: line 1366
   - Q1 adjustment logic: lines 1344-1361
   - `_detect_fiscal_year_end()`: lines 204-224

2. **`src/investigator/domain/services/fiscal_period_service.py`** (430 lines)
   - `detect_fiscal_year_end()` method: lines 230-301
   - Algorithm for finding FY end from 10-K entries

### Supporting Files
3. **`src/investigator/infrastructure/sec/companyfacts_extractor.py`**
   - Fetches raw CompanyFacts API data
   - Source of entry['end'], entry['fy'], entry['fp'], etc.

4. **`src/investigator/infrastructure/sec/canonical_mapper.py`**
   - Extracts metrics using the fiscal_year value

---

## How to Use This Documentation

### For Understanding the Bug (5-10 minutes)
1. Start with **FISCAL_YEAR_QUICK_REFERENCE.txt**
2. Read the data flow section
3. Review the key files and lines

### For Implementation (15-20 minutes)
1. Read **FISCAL_YEAR_INITIAL_ASSIGNMENT_SUMMARY.md** (Sections 1-6)
2. Refer to **FISCAL_YEAR_CODE_LOCATIONS.md** (Location 5: where fiscal_year_end is retrieved)
3. Review the fix at the end of each document

### For Code Review (20-30 minutes)
1. Use **FISCAL_YEAR_CODE_LOCATIONS.md** as reference
2. Cross-check line numbers in data_processor.py
3. Verify fiscal_period_service.py implementation
4. Review the proposed fix logic

### For Architecture Understanding (30+ minutes)
1. Read **FISCAL_YEAR_ASSIGNMENT_TRACE.md** for high-level overview
2. Read **FISCAL_YEAR_INITIAL_ASSIGNMENT_SUMMARY.md** for detailed sections
3. Review **FISCAL_YEAR_CODE_LOCATIONS.md** for implementation details
4. Trace through a specific example (e.g., ORCL Q2)

---

## Key Concepts

### Fiscal Year End (FY End)
- Format: "-MM-DD" (e.g., "-05-31" for ORCL)
- Detected from 10-K filings using `detect_fiscal_year_end()`
- Used to determine if a period belongs to the next fiscal year

### Calendar Year vs Fiscal Year
- **Calendar Year**: Jan 1 - Dec 31 (fiscal_year_end = "-12-31")
- **Non-Calendar Year**: e.g., June 1 - May 31 (fiscal_year_end = "-05-31")
- Bug occurs because code assumes calendar year for all companies

### Period End Date
- Format: "YYYY-MM-DD" (e.g., "2024-11-30")
- Extracted from CompanyFacts API
- Used to derive fiscal_year using `period_end_date.year`

### Fiscal Period
- Format: Q1, Q2, Q3, Q4, or FY
- Derived from period end date month or duration
- Used with fiscal_year to uniquely identify a reporting period

---

## Examples

### ORCL Q2-2025 (THE BUG)
```
Period end: 2024-11-30
FY end: -05-31
Expected fiscal_year: 2025 (because Nov 30 > May 31)
Actual fiscal_year: 2024 (because period_end_date.year = 2024)
Q1 Fix applies? NO (only applies to Q1)
Result: WRONG fiscal_year stored in database
```

### AAPL Q2-2025 (WORKS CORRECTLY)
```
Period end: 2024-12-31 (approximately)
FY end: -09-30
Expected fiscal_year: 2025 (because Dec 31 > Sep 30)
Actual fiscal_year: 2024 (because period_end_date.year = 2024)
Q1 Fix applies? NO (only applies to Q1)
BUT: Works by accident because fiscal_year=2024 happens to be correct
     (AAPL's calendar year matches their fiscal year adjustment)
```

### ZS Q1-2024 (THE Q1 FIX)
```
Period end: 2023-10-31
FY end: -07-31
Expected fiscal_year: 2024 (because Oct 31 > Jul 31)
Actual fiscal_year: 2023 (because period_end_date.year = 2023)
Q1 Fix applies? YES (actual_fp = 'Q1')
Result: fiscal_year adjusted to 2024 (CORRECT)
```

---

## Recommended Reading Order

### Quick Understanding (10 minutes)
1. FISCAL_YEAR_QUICK_REFERENCE.txt

### Implementation (30 minutes)
1. FISCAL_YEAR_QUICK_REFERENCE.txt
2. FISCAL_YEAR_INITIAL_ASSIGNMENT_SUMMARY.md (Key Code Sections)
3. FISCAL_YEAR_CODE_LOCATIONS.md (Location 5)

### Complete Trace (60 minutes)
1. FISCAL_YEAR_QUICK_REFERENCE.txt
2. FISCAL_YEAR_ASSIGNMENT_TRACE.md
3. FISCAL_YEAR_INITIAL_ASSIGNMENT_SUMMARY.md
4. FISCAL_YEAR_CODE_LOCATIONS.md

---

## Related Issues

- ORCL Q2-2025: fiscal_year=2024 (should be 2025)
- ZS Q1-2024: fiscal_year=2023 (fixed by Q1 adjustment)
- Any non-calendar FY company with Q2-Q4 periods will have the same bug

---

## Questions?

Refer to:
- **"Where is fiscal_year set?"** → FISCAL_YEAR_INITIAL_ASSIGNMENT_SUMMARY.md (Section 3)
- **"What's the data flow?"** → FISCAL_YEAR_ASSIGNMENT_TRACE.md (Data Flow section)
- **"Show me the code!"** → FISCAL_YEAR_CODE_LOCATIONS.md
- **"How do I fix it?"** → Any document's "The Fix Required" section

---

**Last Updated**: November 16, 2025
**Status**: Complete documentation set ready for implementation
**Total Lines**: 1,016 lines across 4 documents
