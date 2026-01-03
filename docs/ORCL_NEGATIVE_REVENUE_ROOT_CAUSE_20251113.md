# ORCL Negative Revenue - Root Cause Analysis
**Date**: 2025-11-13
**Symbol**: ORCL (Oracle Corporation)
**Fiscal Year**: May 31 (non-calendar year-end)
**Issue**: Negative revenue in processed table, impossible Q4 computations

---

## Executive Summary

**Problem**: ORCL shows negative revenue in sec_companyfacts_processed table for Q2 periods:
- 2023-Q2: **-$1,085M** (should be ~$12B)
- 2024-Q2: **-$178M** (should be ~$13B)

**Root Cause**: Missing Q2 period data → YTD conversion failure → Corrupted quarterly values stored in processed table → Cascading failures in Q4 computation.

**Impact**:
- ❌ DCF valuation unreliable ($49 vs $227 actual - 78% undervalued but data is corrupt)
- ❌ Revenue growth calculations invalid (-13.9% due to corrupt Q2 data)
- ❌ Quarterly trend analysis broken

---

## Data Flow Analysis

### Fiscal Period Sequence (from logs)
```
['Q2', 'Q3', 'FY', 'Q1', 'Q2', 'Q3', 'FY', 'Q1', 'Q2', 'Q3', 'FY', 'Q1', 'Q3', 'FY', 'Q1']
```

**Grouped by Fiscal Year**:
```
FY 2026: Q1
FY 2025: Q1, Q3 (⚠️  MISSING Q2!)
FY 2024: Q1, Q2, Q3
FY 2023: Q1, Q2, Q3
FY 2022: Q2, Q3
```

**Oracle Fiscal Year**: Ends May 31
- Q1: Jun-Aug (3 months)
- Q2: Sep-Nov (6 months YTD)
- Q3: Dec-Feb (9 months YTD)
- Q4: Mar-May (12 months = FY)

---

## Problem Chain

### 1. Missing Q2 Period
**FY 2025** sequence: `Q1 → Q3` (gap!)

**Why this breaks**:
- Q3 data from SEC is **Year-To-Date (YTD)** = Q1+Q2+Q3 cumulative
- YTD-to-quarterly conversion formula: `Q3 = Q3_YTD - (Q1 + Q2)`
- **Without Q2**: Cannot convert Q3_YTD to actual Q3 value
- Q3 remains unconverted (still has YTD values)

**Log Evidence**:
```
⚠️  Q4 computation SKIPPED for FY 2025: YTD data detected in Q3
```

### 2. Failed YTD Conversion Creates Corrupt Data

**Expected Q2 revenue** (from SEC CompanyFacts for Oracle):
- 2024-Q2 ending 2024-11-30: ~$13B (typical for Oracle Q2)
- 2023-Q2 ending 2023-11-30: ~$12B

**What happened**:
1. Previous pipeline run attempted YTD conversion
2. Q2 data missing or had issues
3. Conversion produced negative/corrupt values
4. Corrupt values stored in `sec_companyfacts_processed` table
5. Future runs use corrupt cached data

**Stored corrupt values**:
- 2023-Q2: **-$1,085,000,000** (Revenue field in processed table)
- 2024-Q2: **-$178,000,000** (Revenue field in processed table)

### 3. Q4 Computation with Corrupt Q1-Q3 Sum

**Mathematically Impossible**:
```
FY 2023:
  FY Total Operating Cash Flow: $17,165M
  Q1+Q2+Q3 Sum: $18,153M  (⚠️  GREATER than FY total!)
  Computed Q4: $17,165M - $18,153M = -$988M (NEGATIVE)
```

**Why Q1-Q3 sum > FY**:
- Q2 has corrupt negative revenue from failed YTD conversion
- But cash flow metrics from Q2 may still be positive
- Creates mismatched data where some metrics are corrupt, others valid
- Summation produces values that exceed fiscal year totals

**Log Evidence**:
```
Computed negative Q4 operating_cash_flow: -988.0M (FY: 17165.0M, Q1-Q3 sum: 18153.0M)
Computed negative Q4 free_cash_flow: -3111.0M (FY: 8470.0M, Q1-Q3 sum: 11581.0M)
```

### 4. Cascading Impact on DCF Valuation

**Revenue Growth Calculation**:
```
Current TTM: $54.93B
Prior1 TTM:  $63.83B
Growth: -13.9%  (⚠️  WRONG - Oracle revenue is actually growing)
```

**Why it's wrong**:
- TTM includes corrupt Q2 periods with negative revenue
- Drags down total revenue incorrectly
- Shows declining revenue when Oracle is actually growing

**DCF Result**:
```
Fair Value: $49.18
Current Price: $226.99
Upside: -78.3%  (massive undervaluation)
```

**Why DCF is unreliable**:
- Based on corrupt revenue growth (-13.9%)
- Negative Q4 FCF periods distort historical FCF growth
- Growth detection sees "turnaround" due to data corruption
- Fallback to conservative 5% growth (should be 8-10% for Oracle)

---

## Root Cause Identification

### Why is Q2 Missing?

**Hypothesis 1: SEC Filing Pattern**
- Oracle may file 10-Q for Q1 and Q3, but not Q2
- Some companies skip Q2 10-Q filings if not material
- Q2 data embedded in Q3 YTD filing only

**Hypothesis 2: Bulk Table Loading Issue**
- Q2 data exists in SEC but not loaded into `sec_num_data` bulk table
- Data processor filters missed Oracle Q2 periods
- Non-calendar fiscal year edge case

**Hypothesis 3: Processed Table Corruption**
- Previous run had Q2 data
- YTD conversion bug created negative values
- Negative values stored in processed table
- Current runs detect "zero/missing revenue" and fall back
- But fallback to bulk tables ALSO missing Q2

**Most Likely**: Combination of #1 and #3
- Q2 period may not exist as standalone 10-Q filing
- Previous pipeline attempted to derive Q2 from Q3_YTD - Q1
- Derivation produced negative values (bug in subtraction logic?)
- Corrupt Q2 stored in processed table

---

## Verification Steps

### 1. Check SEC EDGAR for Oracle Q2 Filings

```bash
# Query bulk tables for ORCL Q2 periods
PGPASSWORD=investigator psql -h ${DB_HOST:-localhost} -U investigator -d sec_database <<EOF
SELECT adsh, cik, form, fp, fy, filed, period
FROM sec_sub_data
WHERE cik = '0000001341'  -- Oracle CIK
  AND fp = 'Q2'
  AND fy IN (2023, 2024, 2025)
ORDER BY filed DESC;
EOF
```

**Expected**: Should return Q2 filings OR empty (confirming Q2 not filed)

### 2. Check Processed Table for Corrupt Data

```bash
PGPASSWORD=investigator psql -h ${DB_HOST:-localhost} -U investigator -d sec_database <<EOF
SELECT fiscal_year, fiscal_period, period_end_date,
       income_statement->>'total_revenue' as revenue,
       income_statement->>'operating_income' as operating_income
FROM sec_companyfacts_processed
WHERE symbol = 'ORCL'
  AND fiscal_period = 'Q2'
ORDER BY period_end_date DESC;
EOF
```

**Expected**: Will show negative revenue values confirming corruption

### 3. Check CompanyFacts API Raw Data

```bash
curl "https://data.sec.gov/api/xbrl/companyfacts/CIK0000001341.json" \
  -H "User-Agent: InvestiGator/1.0" \
  | jq '.facts["us-gaap"].Revenues.units.USD[] | select(.fp == "Q2") | {fy, fp, filed, val}'
```

**Expected**: Should show actual Q2 revenue values (~$12-13B)

---

## Proposed Fixes

### Fix 1: Detect and Skip Corrupt Processed Data

**File**: `src/investigator/domain/agents/fundamental/agent.py`

**Location**: Around line 910 (where processed data validation happens)

**Current Code**:
```python
if quarterly_data['income_statement'].get('total_revenue', 0) <= 0:
    logger.warning(f"⚠️  Processed data for {symbol} {fiscal_period} has zero/missing revenue")
    # Falls back to bulk tables
```

**Problem**: Fallback to bulk tables ALSO has missing Q2, so it doesn't fix the issue.

**Proposed Fix**:
```python
revenue = quarterly_data['income_statement'].get('total_revenue', 0)

# CRITICAL: Detect corrupt negative revenue (data quality issue)
if revenue < 0:
    logger.error(f"❌ CORRUPT DATA: {symbol} {fiscal_period} has NEGATIVE revenue: ${revenue:,.0f}. "
                 f"This indicates failed YTD conversion. DELETING corrupt record.")
    # Delete corrupt record from processed table
    delete_processed_data(symbol, fiscal_year, fiscal_period)
    # Force re-fetch from CompanyFacts API
    return None  # Trigger fresh fetch

if revenue == 0:
    logger.warning(f"⚠️  Processed data for {symbol} {fiscal_period} has zero revenue")
    # Fall back to bulk tables or CompanyFacts
```

### Fix 2: Improve YTD Conversion Logic for Missing Quarters

**File**: `utils/quarterly_calculator.py`

**Function**: `convert_ytd_to_quarterly()`

**Current Issue**: When Q2 is missing, Q3 YTD conversion fails silently, leaving Q3 unconverted.

**Proposed Enhancement**:
```python
def convert_ytd_to_quarterly(periods_by_fy):
    """Convert YTD periods to quarterly with missing period detection"""

    for fiscal_year, quarters in periods_by_fy.items():
        ytd_quarters = [q for q in quarters if q['data_ytd'] == True]

        for ytd_q in ytd_quarters:
            period_num = parse_period_num(ytd_q['fiscal_period'])  # Q1=1, Q2=2, Q3=3

            # Get all prior quarters needed for subtraction
            prior_quarters = [q for q in quarters if parse_period_num(q['fiscal_period']) < period_num]

            # CRITICAL CHECK: Ensure all prior quarters present
            expected_prior = list(range(1, period_num))  # Q3 expects [1, 2]
            actual_prior = [parse_period_num(q['fiscal_period']) for q in prior_quarters]

            missing = set(expected_prior) - set(actual_prior)
            if missing:
                logger.error(f"❌ YTD CONVERSION FAILED: {fiscal_year}-Q{period_num} missing prior quarters: {missing}")
                logger.error(f"   Cannot convert YTD to quarterly. SKIPPING this period to avoid corrupt data.")
                ytd_q['conversion_skipped'] = True
                ytd_q['conversion_error'] = f"Missing Q{missing}"
                continue  # Skip conversion, leave as YTD

            # Perform subtraction only if all priors present
            for metric in FINANCIAL_METRICS:
                ytd_value = ytd_q['income_statement'].get(metric, 0)
                prior_sum = sum(q['income_statement'].get(metric, 0) for q in prior_quarters)
                quarterly_value = ytd_value - prior_sum

                # Sanity check: Negative values indicate data corruption
                if quarterly_value < 0 and metric in MUST_BE_POSITIVE:
                    logger.warning(f"⚠️  Negative {metric} after YTD conversion: {quarterly_value:,.0f} "
                                   f"(YTD: {ytd_value:,.0f}, Prior sum: {prior_sum:,.0f})")

                ytd_q['income_statement'][metric] = quarterly_value

            ytd_q['data_ytd'] = False  # Mark as converted
```

### Fix 3: Add Processed Table Data Quality Check

**File**: New utility `utils/data_quality_checker.py`

```python
def detect_and_fix_corrupt_processed_data(symbol: str):
    """
    Detect and delete corrupt quarterly data from processed table.

    Corruption patterns:
    1. Negative revenue (physically impossible for most companies)
    2. Revenue = $0 when other metrics present (incomplete YTD conversion)
    3. Q1+Q2+Q3 sum > FY total (YTD conversion bug)
    """

    from investigator.infrastructure.database.db import get_engine

    engine = get_engine()

    with engine.connect() as conn:
        # Find negative revenue records
        result = conn.execute(text("""
            SELECT fiscal_year, fiscal_period, period_end_date,
                   income_statement->>'total_revenue' as revenue
            FROM sec_companyfacts_processed
            WHERE symbol = :symbol
              AND (income_statement->>'total_revenue')::numeric < 0
        """), {"symbol": symbol})

        corrupt_records = result.fetchall()

        if corrupt_records:
            logger.error(f"❌ Found {len(corrupt_records)} corrupt records for {symbol}")
            for record in corrupt_records:
                logger.error(f"   {record.fiscal_year}-{record.fiscal_period}: Revenue = ${record.revenue}")

            # Delete corrupt records
            conn.execute(text("""
                DELETE FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                  AND (income_statement->>'total_revenue')::numeric < 0
            """), {"symbol": symbol})

            conn.commit()
            logger.info(f"✅ Deleted {len(corrupt_records)} corrupt records for {symbol}")

        return len(corrupt_records)
```

### Fix 4: Force Re-fetch from CompanyFacts API

**When processed table is corrupt**, bypass it entirely and fetch fresh from SEC:

```python
# In fundamental agent
if detect_corrupt_data(symbol):
    logger.warning(f"Corrupt processed data detected for {symbol}. Bypassing processed table.")
    # Force fresh fetch from CompanyFacts API
    company_facts = await fetch_company_facts_api(symbol, cik)
    # Reprocess with fixed YTD conversion logic
    quarterly_data = process_company_facts(company_facts)
    # Store clean data back to processed table
    store_processed_data(symbol, quarterly_data)
```

---

## Immediate Actions

1. **[ ] Delete corrupt ORCL processed data**:
   ```sql
   DELETE FROM sec_companyfacts_processed
   WHERE symbol = 'ORCL'
     AND fiscal_period = 'Q2'
     AND (income_statement->>'total_revenue')::numeric < 0;
   ```

2. **[ ] Re-run ORCL analysis with --force-refresh**:
   ```bash
   python3 cli_orchestrator.py analyze ORCL --mode standard --force-refresh
   ```

3. **[ ] Verify SEC EDGAR for Q2 filings**:
   - Check if Oracle files standalone Q2 10-Q
   - Or if Q2 data only in Q3 YTD filing

4. **[ ] Implement YTD conversion safeguards**:
   - Add missing quarter detection
   - Skip conversion if priors missing
   - Log errors instead of storing corrupt data

5. **[ ] Add data quality checks**:
   - Negative revenue detection
   - Q1-Q3 sum vs FY total validation
   - Impossible metric combinations

---

## Long-Term Improvements

### 1. Quarterly Data Derivation Logic
For companies that don't file Q2 separately:
```python
# If Q2 missing but Q1 and Q3_YTD available:
Q2 = Q3_YTD - Q1
```

**Current bug**: This derivation may be producing negative values. Need to:
1. Add validation that Q3_YTD > Q1 (sanity check)
2. Log warning if derivation produces negative values
3. Skip storing derived Q2 if invalid

### 2. Processed Table Schema Enhancement
Add corruption detection fields:
```sql
ALTER TABLE sec_companyfacts_processed ADD COLUMN data_quality_score FLOAT;
ALTER TABLE sec_companyfacts_processed ADD COLUMN data_quality_issues TEXT[];
ALTER TABLE sec_companyfacts_processed ADD COLUMN derived_quarters TEXT[];
```

Example:
```json
{
  "data_quality_score": 0.75,
  "data_quality_issues": ["q2_derived", "negative_capex_detected"],
  "derived_quarters": ["Q2"]
}
```

### 3. Automated Data Quality Pipeline
```python
# Run nightly
def data_quality_scan():
    symbols = get_all_symbols()
    for symbol in symbols:
        issues = []

        # Check 1: Negative revenue
        if has_negative_revenue(symbol):
            issues.append("negative_revenue")
            fix_negative_revenue(symbol)

        # Check 2: Q1-Q3 sum > FY
        if quarterly_sum_exceeds_fy(symbol):
            issues.append("quarterly_sum_exceeds_fy")
            recompute_quarters(symbol)

        # Check 3: Missing expected quarters
        if has_missing_quarters(symbol):
            issues.append("missing_quarters")
            attempt_derivation(symbol)

        if issues:
            log_data_quality_issue(symbol, issues)
```

---

## Testing Plan

### Test Case 1: ORCL Q2 Derivation
**Setup**: Delete ORCL Q2 from processed table

**Steps**:
1. Fetch Q1 and Q3_YTD from CompanyFacts
2. Derive Q2 = Q3_YTD - Q1
3. Validate Q2 revenue > 0
4. Store in processed table

**Expected**: Q2 revenue ~$12-13B (positive)

### Test Case 2: Missing Quarter Detection
**Setup**: Company with Q1, Q3 (no Q2)

**Steps**:
1. Attempt YTD conversion on Q3
2. Detect missing Q2
3. Skip conversion, log error
4. Leave Q3 as YTD (do not corrupt data)

**Expected**: Q3 skipped with clear error message

### Test Case 3: Negative Revenue Detection
**Setup**: Processed table with negative revenue

**Steps**:
1. Run data quality check
2. Detect negative revenue
3. Delete corrupt record
4. Re-fetch from API

**Expected**: Clean data after re-fetch

---

## Conclusion

**Root Cause**: Missing Q2 data → Failed YTD conversion → Corrupt negative revenue stored → Cascading Q4 computation failures

**Priority Fixes**:
1. Delete corrupt ORCL processed data (immediate)
2. Add negative revenue detection (prevent future corruption)
3. Improve YTD conversion with missing quarter checks (safeguard)
4. Implement quarterly data derivation logic (long-term)

**Impact**: Once fixed, ORCL DCF should show realistic valuation (likely $180-250 range vs current corrupt $49).
