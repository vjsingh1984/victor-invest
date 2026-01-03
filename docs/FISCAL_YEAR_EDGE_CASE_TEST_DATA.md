# Fiscal Year Edge Case Test Data

**Date**: 2025-11-17
**Purpose**: Reference data for validating fiscal year calculation logic

---

## Real-World Test Cases from Database

### 1. Oracle Corporation (ORCL) - May 31 Fiscal Year End

**CIK**: 1341439
**Fiscal Year End**: 0531 (May 31)

| Period End   | FY   | FP  | Filed      | Period Month | Period Year | Notes                          |
|--------------|------|-----|------------|--------------|-------------|--------------------------------|
| 2025-05-31   | 2025 | FY  | 2025-06-18 | 5            | 2025        | FY period, May == May → FY 2025|
| 2025-02-28   | 2025 | Q3  | 2025-03-11 | 2            | 2025        | Feb < May → FY 2025            |
| 2024-11-30   | 2025 | Q2  | 2024-12-10 | 11           | 2024        | Nov > May → FY 2025            |
| 2024-08-31   | 2025 | Q1  | 2024-09-10 | 8            | 2024        | Aug > May → FY 2025            |
| 2024-05-31   | 2024 | FY  | 2024-06-20 | 5            | 2024        | FY period, May == May → FY 2024|
| 2024-02-29   | 2024 | Q3  | 2024-03-12 | 2            | 2024        | Leap year, Feb < May → FY 2024 |
| 2023-11-30   | 2024 | Q2  | 2023-12-12 | 11           | 2023        | Nov > May → FY 2024            |
| 2023-08-31   | 2024 | Q1  | 2023-09-12 | 8            | 2023        | Aug > May → FY 2024            |

**Key Insight**: For May FYE (2nd half of year), when `period_month == fiscal_year_end_month`, `fiscal_year = period_year`

---

### 2. Walmart Inc. (WMT) - January 31 Fiscal Year End

**CIK**: 310354
**Fiscal Year End**: 0131 (January 31)

| Period End   | FY   | FP  | Filed      | Period Month | Period Year | Notes                              |
|--------------|------|-----|------------|--------------|-------------|------------------------------------|
| 2025-04-30   | 2026 | Q1  | 2025-06-06 | 4            | 2025        | Apr > Jan → FY 2026                |
| 2025-01-31   | 2024 | FY  | 2025-03-14 | 1            | 2025        | **FY period, Jan == Jan → FY 2024**|
| 2024-10-31   | 2025 | Q3  | 2024-12-06 | 10           | 2024        | Oct > Jan → FY 2025                |
| 2024-07-31   | 2025 | Q2  | 2024-08-30 | 7            | 2024        | Jul > Jan → FY 2025                |
| 2024-04-30   | 2025 | Q1  | 2024-06-07 | 4            | 2024        | Apr > Jan → FY 2025                |
| 2024-01-31   | 2023 | FY  | 2024-03-15 | 1            | 2024        | **FY period, Jan == Jan → FY 2023**|
| 2023-10-31   | 2024 | Q3  | 2023-11-30 | 10           | 2023        | Oct > Jan → FY 2024                |
| 2023-07-31   | 2024 | Q2  | 2023-09-01 | 7            | 2023        | Jul > Jan → FY 2024                |

**Key Insight**: For Jan FYE (1st half of year), when `period_month == fiscal_year_end_month`, `fiscal_year = period_year - 1` ⚠️

---

### 3. Microsoft Corporation (MSFT) - June 30 Fiscal Year End

**Fiscal Year End**: 0630 (June 30)

| Period End   | FY   | FP  | Period Month | Period Year | Notes                          |
|--------------|------|-----|--------------|-------------|--------------------------------|
| 2024-06-30   | 2024 | FY  | 6            | 2024        | FY period, Jun == Jun → FY 2024|
| 2023-06-30   | 2023 | FY  | 6            | 2023        | FY period, Jun == Jun → FY 2023|
| 2022-06-30   | 2022 | FY  | 6            | 2022        | FY period, Jun == Jun → FY 2022|

**Key Insight**: For Jun FYE (2nd half of year), when `period_month == fiscal_year_end_month`, `fiscal_year = period_year`

---

### 4. Amazon.com Inc. (AMZN) - December 31 Fiscal Year End (Calendar Year)

**Fiscal Year End**: 1231 (December 31)

| Period End   | FY   | FP  | Period Month | Period Year | Notes                          |
|--------------|------|-----|--------------|-------------|--------------------------------|
| 2024-12-31   | 2024 | FY  | 12           | 2024        | Calendar year                  |
| 2023-12-31   | 2023 | FY  | 12           | 2023        | Calendar year                  |
| 2022-12-31   | 2022 | FY  | 12           | 2022        | Calendar year                  |

**Key Insight**: Calendar year companies (Dec FYE), `fiscal_year = period_year`

---

### 5. Companies with February Fiscal Year End

**Fiscal Year End**: 0228 (February 28)

| Period End   | FY   | FP  | Period Month | Period Year | Notes                              |
|--------------|------|-----|--------------|-------------|------------------------------------|
| 2025-02-28   | 2024 | FY  | 2            | 2025        | **FY period, Feb == Feb → FY 2024**|

**Key Insight**: For Feb FYE (1st half of year), when `period_month == fiscal_year_end_month`, `fiscal_year = period_year - 1` ⚠️

---

## Edge Case Pattern Summary

### Rule 1: Period Month > Fiscal Year End Month
**Always**: `fiscal_year = period_year + 1`

**Examples**:
- ORCL (May FYE): Aug 2024 → FY 2025 ✓
- WMT (Jan FYE): Apr 2024 → FY 2025 ✓

### Rule 2: Period Month == Fiscal Year End Month (CRITICAL EDGE CASE)

**If FYE in Jan-Jun (months 1-6)**:
- `fiscal_year = period_year - 1`
- **Reason**: Fiscal year LABEL equals calendar year of the START of the fiscal year

**Examples**:
- WMT (Jan FYE): Jan 31, 2025 → FY 2024 ✓
- Feb FYE: Feb 28, 2025 → FY 2024 ✓

**If FYE in Jul-Dec (months 7-12)**:
- `fiscal_year = period_year`
- **Reason**: Fiscal year LABEL equals calendar year of the END of the fiscal year

**Examples**:
- ORCL (May FYE): May 31, 2025 → FY 2025 ✓
- MSFT (Jun FYE): Jun 30, 2024 → FY 2024 ✓
- AMZN (Dec FYE): Dec 31, 2024 → FY 2024 ✓

### Rule 3: Period Month < Fiscal Year End Month
**Always**: `fiscal_year = period_year`

**Examples**:
- ORCL (May FYE): Feb 2025 → FY 2025 ✓
- WMT (Jan FYE): (Not applicable - all months > Jan) N/A

---

## Leap Year Edge Cases

### American Greetings Corporation - February 29 Fiscal Year End

**CIK**: 5133
**Fiscal Year End**: 0229 (February 29 in leap years, February 28 otherwise)

| Period End   | FY   | FP  | Filed      | FYE  | Notes                    |
|--------------|------|-----|------------|------|--------------------------|
| 2016-02-29   | 2015 | FY  | 2016-05-26 | 0229 | Leap year FY             |
| 2015-02-28   | 2014 | FY  | 2015-05-15 | 0228 | Non-leap year FY         |

**Key Insight**: Leap year companies handle Feb 29/28 correctly in bulk tables

---

## Validation Test Matrix

### Fiscal Year Calculation Tests

| Period End   | FY End Month | Expected FY | Calculated FY | Status | Notes                    |
|--------------|--------------|-------------|---------------|--------|--------------------------|
| 2024-08-31   | 5 (May)      | 2025        | 2025          | PASS   | Aug > May                |
| 2024-11-30   | 5 (May)      | 2025        | 2025          | PASS   | Nov > May                |
| 2024-05-31   | 5 (May)      | 2024        | 2024          | PASS   | May == May (2nd half)    |
| 2024-02-29   | 5 (May)      | 2024        | 2024          | PASS   | Leap year, Feb < May     |
| 2025-04-30   | 1 (Jan)      | 2026        | 2026          | PASS   | Apr > Jan                |
| 2024-10-31   | 1 (Jan)      | 2025        | 2025          | PASS   | Oct > Jan                |
| 2025-01-31   | 1 (Jan)      | 2024        | 2025          | FAIL   | Jan == Jan (1st half)    |
| 2024-01-31   | 1 (Jan)      | 2023        | 2024          | FAIL   | Jan == Jan (1st half)    |
| 2025-02-28   | 2 (Feb)      | 2024        | 2025          | FAIL   | Feb == Feb (1st half)    |
| 2024-06-30   | 6 (Jun)      | 2024        | 2024          | PASS   | Jun == Jun (2nd half)    |
| 2024-12-31   | 12 (Dec)     | 2024        | 2024          | PASS   | Calendar year            |

**Summary**: 8/11 PASS before fix, 11/11 PASS after fix

---

## Database Queries for Validation

### Query 1: Check Fiscal Year End Distribution
```sql
SELECT DISTINCT fye, COUNT(DISTINCT cik) as company_count
FROM sec_sub_data
WHERE fye IS NOT NULL
GROUP BY fye
ORDER BY company_count DESC, fye;
```

### Query 2: Verify Company Fiscal Periods
```sql
-- Oracle (May FYE)
SELECT period, fy, fp, filed, fye,
  EXTRACT(MONTH FROM period) as period_month,
  EXTRACT(YEAR FROM period) as period_year
FROM sec_sub_data
WHERE cik = 1341439  -- Oracle
  AND form IN ('10-K', '10-Q')
  AND fy >= 2023
ORDER BY period DESC;
```

### Query 3: Find Edge Case Companies
```sql
-- Find companies with Jan-Jun fiscal year ends
SELECT fye, COUNT(DISTINCT cik) as count
FROM sec_sub_data
WHERE fye IN ('0131', '0228', '0229', '0331', '0430', '0531', '0630')
GROUP BY fye
ORDER BY fye;
```

### Query 4: Validate Fiscal Year Calculation
```sql
-- Compare processed fiscal_year vs bulk table fiscal_year
SELECT
  proc.symbol,
  proc.fiscal_year as processed_fy,
  proc.fiscal_period as processed_fp,
  sub.fy as bulk_fy,
  sub.fp as bulk_fp,
  proc.period_end_date,
  sub.fye,
  CASE WHEN proc.fiscal_year = sub.fy THEN 'MATCH' ELSE 'MISMATCH' END as status
FROM sec_companyfacts_processed proc
JOIN sec_sub_data sub ON proc.adsh = sub.adsh
WHERE proc.symbol IN ('ORCL', 'WMT', 'MSFT', 'AMZN')
  AND proc.fiscal_period = 'FY'
  AND sub.form = '10-K'
ORDER BY proc.symbol, proc.fiscal_year DESC
LIMIT 20;
```

---

## Test Execution

### Run Edge Case Validation
```bash
cd /Users/vijaysingh/code/InvestiGator
python3 tests/edge_case_fiscal_year_validation.py
```

**Expected Output** (after fix):
```
================================================================================
OVERALL SUMMARY
================================================================================
Total Tests: 28
Passed: 28 (100.0%)
Failed: 0 (0.0%)
Errors: 0 (0.0%)

✓ ALL VALIDATIONS PASSED!
```

---

## Companies Affected by Edge Case

### High-Risk Companies (FYE in Jan-Jun)

| FYE Code | Month | Count | % of DB | Examples         |
|----------|-------|-------|---------|------------------|
| 0131     | Jan   | 392   | 2.2%    | WMT              |
| 0228     | Feb   | 205   | 1.1%    | Various          |
| 0229     | Feb*  | 160   | 0.9%    | Various          |
| 0331     | Mar   | 930   | 5.2%    | Various          |
| 0430     | Apr   | 252   | 1.4%    | Various          |
| 0531     | May   | 237   | 1.3%    | ORCL             |
| 0630     | Jun   | 1037  | 5.8%    | MSFT, STX        |

**Total**: ~3,213 companies (18% of database)

**Risk**: Without fix, FY periods where `period_month == fiscal_year_end_month` will have **incorrect fiscal year labels** for Jan-Jun FYE companies.

---

## Conclusion

This document provides comprehensive test data from the SEC database to validate fiscal year calculation logic across all edge cases. The **critical finding** is that companies with fiscal year ends in **Jan-Jun** require special handling when `period_month == fiscal_year_end_month`, where `fiscal_year = period_year - 1` instead of `period_year`.
