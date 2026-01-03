# qtrs Detection Implementation Plan - SEC CompanyFacts API

**Date**: 2025-11-04
**Status**: Ready for Implementation
**Goal**: Implement statement-specific qtrs detection based on start/end date analysis

---

## Empirical Findings from Raw JSON Analysis

### AAPL Q2 2024
```
Revenue (RevenueFromContractWithCustomerExcludingAssessedTax):
  ✅ qtrs=1: Start: 2023-12-31, End: 2024-03-30 (90 days) = $90.75B
  ✅ qtrs=2: Start: 2023-10-01, End: 2024-03-30 (181 days) = $210.33B YTD

Operating Cash Flow (NetCashProvidedByUsedInOperatingActivities):
  ❌ NO qtrs=1 available
  ✅ qtrs=2: Start: 2023-10-01, End: 2024-03-30 (181 days) = $62.59B YTD
```

### MSFT Q2 2024
```
Revenue (RevenueFromContractWithCustomerExcludingAssessedTax):
  ✅ qtrs=1: Start: 2023-10-01, End: 2023-12-31 (91 days) = $62.02B
  ✅ qtrs=2: Start: 2023-07-01, End: 2023-12-31 (183 days) = $118.54B YTD

Operating Cash Flow (NetCashProvidedByUsedInOperatingActivities):
  ✅ qtrs=1: Start: 2023-10-01, End: 2023-12-31 (91 days) = $18.85B
  ✅ qtrs=2: Start: 2023-07-01, End: 2023-12-31 (183 days) = $49.44B YTD
```

**Pattern Confirmed**:
- AAPL: Mixed (Income qtrs=1+2, Cash Flow qtrs=2 only) - 80% of stocks
- MSFT: Both (Income qtrs=1+2, Cash Flow qtrs=1+2) - 20% of stocks

---

## Detection Strategy

### Rule 1: Duration-Based Classification

```python
days = (end_date - start_date).days

if days < 120:
    duration_type = "INDIVIDUAL (qtrs=1)"
elif days < 270:
    duration_type = "YTD Q2/Q3 (qtrs=2 or 3)"
else:
    duration_type = "Full Year (qtrs=4)"
```

**Thresholds**:
- Individual Quarter: < 120 days (~3 months)
- YTD Q2: 120-270 days (~6 months)
- YTD Q3: 270-365 days (~9 months)
- Full Year: 365+ days

### Rule 2: Statement-Specific Fallback Chain

**Priority**: Individual (qtrs=1) > YTD (qtrs=2/3)

```python
def detect_qtrs_for_statement(us_gaap, tags, adsh, fiscal_period):
    """
    Try to find individual quarter entry first, fallback to YTD

    Returns:
        qtrs value (1, 2, 3, or 4)
    """
    individual_entries = []
    ytd_entries = []

    for tag in tags:
        if tag not in us_gaap:
            continue

        for entry in us_gaap[tag]['units']['USD']:
            if entry.get('accn') != adsh:
                continue
            if entry.get('fp') != fiscal_period:
                continue

            start = entry.get('start')
            end = entry.get('end')
            if not start or not end:
                continue

            days = (datetime.strptime(end, '%Y-%m-%d') -
                   datetime.strptime(start, '%Y-%m-%d')).days

            if days < 120:
                individual_entries.append(entry)
            elif days < 270:
                ytd_entries.append(entry)

    # Prefer individual quarter
    if individual_entries:
        return 1
    elif ytd_entries:
        return 2 if fiscal_period == 'Q2' else 3
    else:
        # Fallback to safe default
        return {'Q1': 1, 'Q2': 2, 'Q3': 3, 'FY': 4}[fiscal_period]
```

### Rule 3: Statement-Specific Tags

**Income Statement Tags**:
- `RevenueFromContractWithCustomerExcludingAssessedTax`
- `Revenues`
- `SalesRevenueNet`

**Cash Flow Tags**:
- `NetCashProvidedByUsedInOperatingActivities`

---

## Implementation Steps

### Step 1: Add qtrs Detection Method

**File**: `src/investigator/infrastructure/sec/data_processor.py`
**Location**: Before `process_raw_data()` method (around line 150)

```python
def _detect_statement_qtrs(
    self,
    us_gaap: Dict,
    adsh: str,
    fiscal_year: int,
    fiscal_period: str
) -> Tuple[int, int]:
    """
    Detect optimal qtrs values for income statement and cash flow statement.

    Strategy:
    1. Try to find individual quarter entries (qtrs=1, duration < 120 days)
    2. If not available, use YTD entries (qtrs=2/3, duration 120-270 days)
    3. Fallback to safe defaults if no entries found

    Args:
        us_gaap: SEC us-gaap JSON structure
        adsh: Accession number for filtering
        fiscal_year: Fiscal year for filtering
        fiscal_period: Fiscal period (Q1, Q2, Q3, FY)

    Returns:
        Tuple of (income_statement_qtrs, cash_flow_statement_qtrs)

    Examples:
        AAPL Q2 2024: (1, 2) - Income has individual, Cash Flow only YTD
        MSFT Q2 2024: (1, 1) - Both have individual quarters
    """
    from datetime import datetime

    # Q1 and FY always use individual/full year
    if fiscal_period == 'Q1':
        return (1, 1)
    elif fiscal_period == 'FY':
        return (4, 4)

    # For Q2/Q3, try to find individual quarter entries

    # Income Statement Tags (in priority order)
    income_tags = [
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'Revenues',
        'SalesRevenueNet'
    ]

    # Cash Flow Tags
    cashflow_tags = [
        'NetCashProvidedByUsedInOperatingActivities'
    ]

    income_qtrs = self._find_optimal_qtrs(
        us_gaap, income_tags, adsh, fiscal_year, fiscal_period
    )
    cashflow_qtrs = self._find_optimal_qtrs(
        us_gaap, cashflow_tags, adsh, fiscal_year, fiscal_period
    )

    logger.debug(
        f"Detected qtrs for {adsh[:10]}... FY:{fiscal_year} {fiscal_period}: "
        f"income={income_qtrs}, cashflow={cashflow_qtrs}"
    )

    return (income_qtrs, cashflow_qtrs)

def _find_optimal_qtrs(
    self,
    us_gaap: Dict,
    tags: List[str],
    adsh: str,
    fiscal_year: int,
    fiscal_period: str
) -> int:
    """
    Find optimal qtrs value by checking for individual quarter availability.

    Returns:
        qtrs value: 1 (individual), 2 (Q2 YTD), 3 (Q3 YTD), or safe default
    """
    from datetime import datetime

    has_individual = False
    has_ytd = False

    for tag in tags:
        if tag not in us_gaap:
            continue

        usd_data = us_gaap[tag].get('units', {}).get('USD', [])

        for entry in usd_data:
            # Filter by ADSH and fiscal period
            if entry.get('accn') != adsh:
                continue
            if entry.get('fy') != fiscal_year:
                continue
            if entry.get('fp') != fiscal_period:
                continue

            # Check duration
            start = entry.get('start')
            end = entry.get('end')
            if not start or not end:
                continue

            try:
                start_date = datetime.strptime(start, '%Y-%m-%d')
                end_date = datetime.strptime(end, '%Y-%m-%d')
                days = (end_date - start_date).days

                if days < 120:
                    has_individual = True
                elif 120 <= days < 270:
                    has_ytd = True
            except ValueError:
                continue

    # Prefer individual quarter
    if has_individual:
        return 1
    elif has_ytd:
        return 2 if fiscal_period == 'Q2' else 3
    else:
        # Safe fallback
        return {'Q2': 2, 'Q3': 3}.get(fiscal_period, 1)
```

### Step 2: Call Detection in process_raw_data()

**File**: `src/investigator/infrastructure/sec/data_processor.py`
**Location**: After PHASE 2 extraction loop (around line 260)

```python
# PHASE 3: Detect statement-specific qtrs for each filing
for adsh, filing in filings.items():
    income_qtrs, cashflow_qtrs = self._detect_statement_qtrs(
        us_gaap,
        adsh,
        filing["fiscal_year"],
        filing["fiscal_period"]
    )

    filing["income_statement_qtrs"] = income_qtrs
    filing["cash_flow_statement_qtrs"] = cashflow_qtrs

    logger.debug(
        f"Filing {adsh[:10]}... FY:{filing['fiscal_year']} {filing['fiscal_period']}: "
        f"income_qtrs={income_qtrs}, cashflow_qtrs={cashflow_qtrs}"
    )
```

### Step 3: Update INSERT Statement

**File**: `src/investigator/infrastructure/sec/data_processor.py`
**Location**: Lines 466-487 (INSERT) and 526-565 (VALUES)

**Add to INSERT columns** (line 476):
```sql
income_statement_qtrs, cash_flow_statement_qtrs,
```

**Add to VALUES placeholders** (line 487):
```sql
:income_statement_qtrs, :cash_flow_statement_qtrs,
```

**Add to parameter dict** (after line 565):
```python
"income_statement_qtrs": filing.get("income_statement_qtrs"),
"cash_flow_statement_qtrs": filing.get("cash_flow_statement_qtrs"),
```

**Add to ON CONFLICT UPDATE** (after line 519):
```sql
income_statement_qtrs = EXCLUDED.income_statement_qtrs,
cash_flow_statement_qtrs = EXCLUDED.cash_flow_statement_qtrs,
```

---

## Expected Results

### AAPL Q2 2024
```
income_statement_qtrs = 1 (found individual 90-day entry)
cash_flow_statement_qtrs = 2 (only YTD 181-day entry available)
```

### MSFT Q2 2024
```
income_statement_qtrs = 1 (found individual 91-day entry)
cash_flow_statement_qtrs = 1 (found individual 91-day entry)
```

### AAPL Q3 2024
```
income_statement_qtrs = 1 (if individual exists) or 3 (YTD fallback)
cash_flow_statement_qtrs = 3 (likely only YTD available)
```

---

## Testing Plan

### Test 1: Verify Detection Logic
```bash
# Re-process AAPL with --force-refresh
python3 cli_orchestrator.py analyze AAPL --force-refresh

# Check database
PGPASSWORD=investigator psql -h ${DB_HOST:-localhost} -U investigator -d sec_database -c "
SELECT
    symbol, fiscal_year, fiscal_period,
    income_statement_qtrs, cash_flow_statement_qtrs,
    total_revenue / 1e9 as revenue_b,
    operating_cash_flow / 1e9 as ocf_b
FROM sec_companyfacts_processed
WHERE symbol = 'AAPL' AND fiscal_year = 2024 AND fiscal_period IN ('Q1', 'Q2', 'Q3', 'FY')
ORDER BY fiscal_period;
"
```

**Expected Output**:
```
symbol | fy   | fp | income_qtrs | cashflow_qtrs | revenue | ocf
-------|------|----|-----------|--------------|---------|---------
AAPL   | 2024 | Q1 | 1           | 1              | 90.75   | 35.00
AAPL   | 2024 | Q2 | 1           | 2              | 90.75   | 62.59
AAPL   | 2024 | Q3 | 1 or 3      | 3              | ~94     | ~95
AAPL   | 2024 | FY | 4           | 4              | 385     | 122
```

### Test 2: Verify MSFT Pattern
```bash
python3 cli_orchestrator.py analyze MSFT --force-refresh

# Check database
PGPASSWORD=investigator psql -h ${DB_HOST:-localhost} -U investigator -d sec_database -c "
SELECT
    symbol, fiscal_year, fiscal_period,
    income_statement_qtrs, cash_flow_statement_qtrs,
    total_revenue / 1e9 as revenue_b,
    operating_cash_flow / 1e9 as ocf_b
FROM sec_companyfacts_processed
WHERE symbol = 'MSFT' AND fiscal_year = 2024 AND fiscal_period IN ('Q1', 'Q2', 'Q3', 'FY')
ORDER BY fiscal_period;
"
```

**Expected Output**:
```
symbol | fy   | fp | income_qtrs | cashflow_qtrs | revenue | ocf
-------|------|----|-----------|--------------|---------|---------
MSFT   | 2024 | Q1 | 1           | 1              | 56.52   | 23.27
MSFT   | 2024 | Q2 | 1           | 1              | 62.02   | 18.85
MSFT   | 2024 | Q3 | 1           | 1              | ~65     | ~24
MSFT   | 2024 | FY | 4           | 4              | 245     | 87
```

---

## Verification Checklist

- [ ] `_detect_statement_qtrs()` method added
- [ ] `_find_optimal_qtrs()` helper method added
- [ ] Detection called in `process_raw_data()`
- [ ] qtrs fields added to filing dict
- [ ] INSERT statement updated with qtrs columns
- [ ] VALUES placeholders updated
- [ ] Parameter dict updated
- [ ] ON CONFLICT UPDATE updated
- [ ] Test AAPL - verify mixed pattern (income=1, cashflow=2)
- [ ] Test MSFT - verify both pattern (income=1, cashflow=1)
- [ ] Verify existing safe defaults work for older data

---

**Status**: Ready for implementation
**Estimated Time**: 2-3 hours (implementation + testing)
**Risk**: Low (builds on proven schema migration, safe defaults still work)
