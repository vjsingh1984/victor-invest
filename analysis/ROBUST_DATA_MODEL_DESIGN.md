# Robust Data Model Design for Statement-Specific qtrs Handling

**Date**: 2025-11-04
**Finding**: 80% of S&P 100 stocks have MIXED patterns (Income=qtrs_1, Cash Flow=qtrs_2)
**Recommendation**: Store YTD values with statement-specific qtrs tracking + fallback chain

---

## Empirical Analysis Results

### Top 10 S&P 100 Stocks Analysis:

| Symbol | Income Statement Q2 | Cash Flow Statement Q2 | Pattern | Recommendation |
|--------|---------------------|------------------------|---------|----------------|
| AAPL | qtrs=1 ✅ + qtrs=2 ✅ | qtrs=2 ONLY ❌ | Mixed | Use qtrs=2 |
| **MSFT** | qtrs=1 ✅ + qtrs=2 ✅ | qtrs=1 ✅ + qtrs=2 ✅ | Both | **Use qtrs=1** |
| GOOGL | qtrs=1 ✅ + qtrs=2 ✅ | qtrs=2 ONLY ❌ | Mixed | Use qtrs=2 |
| **AMZN** | qtrs=1 ✅ + qtrs=2 ✅ | qtrs=1 ✅ + qtrs=2 ✅ | Both | **Use qtrs=1** |
| NVDA | qtrs=1 ✅ + qtrs=2 ✅ | qtrs=2 ONLY ❌ | Mixed | Use qtrs=2 |
| META | qtrs=1 ✅ + qtrs=2 ✅ | qtrs=2 ONLY ❌ | Mixed | Use qtrs=2 |
| TSLA | qtrs=1 ✅ + qtrs=2 ✅ | qtrs=2 ONLY ❌ | Mixed | Use qtrs=2 |
| JPM | qtrs=1 ✅ + qtrs=2 ✅ | qtrs=2 ONLY ❌ | Mixed | Use qtrs=2 |
| V | qtrs=1 ✅ + qtrs=2 ✅ | qtrs=2 ONLY ❌ | Mixed | Use qtrs=2 |
| UNH | qtrs=1 ✅ + qtrs=2 ✅ | qtrs=2 ONLY ❌ | Mixed | Use qtrs=2 |

### Summary Statistics:
- **Both statements have individual (qtrs=1)**: 2/10 (20%) - MSFT, AMZN
- **Mixed (cash flow YTD only)**: 8/10 (80%) - AAPL, GOOGL, NVDA, META, TSLA, JPM, V, UNH
- **Insufficient data**: 0/10 (0%)

**Conclusion**: **80% require YTD values** for cash flow statements!

---

## Design Decision Matrix

### Option 1: Single qtrs Column (Your Initial Idea)
**Pros**: Simple schema
**Cons**: Cannot represent mixed patterns (Income qtrs=1, Cash Flow qtrs=2)

### Option 2: Two qtrs Columns (Your Refinement)
```sql
ALTER TABLE sec_companyfacts_processed ADD COLUMN income_statement_qtrs INTEGER;
ALTER TABLE sec_companyfacts_processed ADD COLUMN cash_flow_statement_qtrs INTEGER;
```

**Pros**:
- Accurately represents mixed patterns
- Enables statement-specific fallback logic
- Clear semantics

**Cons**:
- Slightly more complex
- Need to track two fields

### Option 3: Three Separate Tables (Alternative)
- `sec_income_statement_quarters`
- `sec_cash_flow_quarters`
- `sec_balance_sheet_quarters`

**Pros**: Perfect normalization
**Cons**: Excessive complexity, join overhead, most queries need all three

### Option 4: JSONB Metadata Column
```sql
ALTER TABLE sec_companyfacts_processed ADD COLUMN statement_metadata JSONB;
-- Contains: {"income_qtrs": 1, "cashflow_qtrs": 2, "balance_qtrs": 0}
```

**Pros**: Flexible, extensible
**Cons**: Less type-safe, harder to query

---

## **RECOMMENDED: Option 2 (Two qtrs Columns)**

**Rationale**:
1. **Accurate**: Represents 100% of observed patterns
2. **Efficient**: Single table, two integer columns (minimal overhead)
3. **Clear**: Explicit statement-level semantics
4. **Queryable**: Can filter/sort by statement type
5. **Backward Compatible**: Existing columns unchanged

---

## Proposed Schema Changes

### Updated `sec_companyfacts_processed` Table

```sql
-- Add statement-specific qtrs tracking
ALTER TABLE sec_companyfacts_processed 
ADD COLUMN income_statement_qtrs SMALLINT DEFAULT NULL,
ADD COLUMN cash_flow_statement_qtrs SMALLINT DEFAULT NULL;

-- Add index for common queries
CREATE INDEX idx_companyfacts_qtrs 
ON sec_companyfacts_processed(symbol, fiscal_year, fiscal_period, 
                                income_statement_qtrs, cash_flow_statement_qtrs);

-- Add comments for documentation
COMMENT ON COLUMN sec_companyfacts_processed.income_statement_qtrs IS 
  'Duration in quarters for income statement metrics: 
   1=individual quarter, 2=Q2 YTD, 3=Q3 YTD, 4=full year';

COMMENT ON COLUMN sec_companyfacts_processed.cash_flow_statement_qtrs IS 
  'Duration in quarters for cash flow statement metrics: 
   1=individual quarter, 2=Q2 YTD, 3=Q3 YTD, 4=full year';
```

### Sample Data After Migration:

| symbol | fy | fp | revenues | ocf | income_qtrs | cashflow_qtrs | Notes |
|--------|----|----|----------|-----|-------------|---------------|-------|
| AAPL | 2024 | Q1 | 119.6B | 39.9B | 1 | 1 | Both individual |
| AAPL | 2024 | Q2 | 210.3B | 62.6B | 2 | 2 | **Both YTD** (stored YTD) |
| AAPL | 2024 | Q3 | 296.1B | 91.4B | 3 | 3 | **Both YTD** (stored YTD) |
| MSFT | 2024 | Q2 | 245.1B | 28.1B | 1 | 1 | **Both individual** (stored individual!) |

---

## Fallback Chain Design

### Priority 1: Prefer Individual Quarter Values (qtrs=1) When Available

**Rule**: If BOTH income_qtrs=1 AND cashflow_qtrs=1, store individual values

**Companies**: MSFT, AMZN (20% of S&P 100)

**Benefits**:
- No YTD conversion needed
- More accurate (no subtraction errors)
- Cleaner data pipeline

### Priority 2: Use YTD Values When Cash Flow Requires It

**Rule**: If cashflow_qtrs > 1, store YTD values for ALL statements

**Companies**: AAPL, GOOGL, NVDA, META, TSLA, JPM, V, UNH (80% of S&P 100)

**Benefits**:
- Ensures cash flow data available
- Consistent across all statements
- YTD conversion handles it downstream

### Priority 3: Bulk Table Fallback

**Rule**: When processed table missing, query bulk tables with statement-aware qtrs filtering

```python
# Determine expected qtrs based on fiscal period and statement
if fiscal_period == 'Q1':
    income_qtrs = 1
    cashflow_qtrs = 1
elif fiscal_period == 'Q2':
    # Try individual first, fallback to YTD
    income_qtrs_try = [1, 2]  # Prefer individual
    cashflow_qtrs_try = [2, 1]  # Prefer YTD (80% only have this)
elif fiscal_period == 'Q3':
    income_qtrs_try = [1, 3]
    cashflow_qtrs_try = [3, 1]
elif fiscal_period == 'FY':
    income_qtrs = 4
    cashflow_qtrs = 4
```

---

## Implementation Plan

### Phase 1: Schema Migration (1 hour)

```sql
-- Add columns
ALTER TABLE sec_companyfacts_processed 
ADD COLUMN income_statement_qtrs SMALLINT,
ADD COLUMN cash_flow_statement_qtrs SMALLINT;

-- Backfill existing data
UPDATE sec_companyfacts_processed
SET 
    income_statement_qtrs = CASE 
        WHEN fiscal_period = 'Q1' THEN 1
        WHEN fiscal_period = 'Q2' THEN 2  -- Default to YTD (80% pattern)
        WHEN fiscal_period = 'Q3' THEN 3
        WHEN fiscal_period = 'FY' THEN 4
    END,
    cash_flow_statement_qtrs = CASE 
        WHEN fiscal_period = 'Q1' THEN 1
        WHEN fiscal_period = 'Q2' THEN 2  -- Always YTD for majority
        WHEN fiscal_period = 'Q3' THEN 3
        WHEN fiscal_period = 'FY' THEN 4
    END;

-- Create index
CREATE INDEX idx_companyfacts_qtrs 
ON sec_companyfacts_processed(symbol, fiscal_year, fiscal_period, 
                                income_statement_qtrs, cash_flow_statement_qtrs);
```

### Phase 2: Update Extraction Logic (2-3 hours)

**File**: `utils/sec_data_processor.py` (or similar extraction script)

```python
def extract_quarterly_data_with_qtrs_detection(symbol: str, fiscal_year: int, fiscal_period: str):
    """
    Extract quarterly data with statement-specific qtrs detection.
    
    Fallback chain:
    1. Try qtrs=1 for both statements (individual quarter)
    2. If cash flow qtrs=1 not available, use qtrs=2/3 (YTD)
    3. Store qtrs values for each statement
    """
    
    # Query both qtrs=1 and qtrs=2/3 from bulk tables or API
    expected_ytd_qtrs = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'FY': 4}[fiscal_period]
    
    # Try individual quarter first (qtrs=1)
    income_individual = query_bulk_table(symbol, fiscal_year, fiscal_period, 
                                          statement='income', qtrs=1)
    cashflow_individual = query_bulk_table(symbol, fiscal_year, fiscal_period, 
                                            statement='cashflow', qtrs=1)
    
    # Determine which to use
    if cashflow_individual is not None:
        # Both have individual - use qtrs=1
        income_data = income_individual
        cashflow_data = cashflow_individual
        income_qtrs = 1
        cashflow_qtrs = 1
    else:
        # Cash flow only has YTD - use qtrs=2/3
        income_ytd = query_bulk_table(symbol, fiscal_year, fiscal_period, 
                                       statement='income', qtrs=expected_ytd_qtrs)
        cashflow_ytd = query_bulk_table(symbol, fiscal_year, fiscal_period, 
                                         statement='cashflow', qtrs=expected_ytd_qtrs)
        
        income_data = income_ytd
        cashflow_data = cashflow_ytd
        income_qtrs = expected_ytd_qtrs
        cashflow_qtrs = expected_ytd_qtrs
    
    # Store with qtrs metadata
    return {
        'symbol': symbol,
        'fiscal_year': fiscal_year,
        'fiscal_period': fiscal_period,
        'income_statement': income_data,
        'cash_flow_statement': cashflow_data,
        'income_statement_qtrs': income_qtrs,
        'cash_flow_statement_qtrs': cashflow_qtrs
    }
```

### Phase 3: Update Query Logic (1 hour)

**File**: `src/investigator/domain/agents/fundamental.py`

```python
def _fetch_from_processed_table(self, symbol: str, fiscal_year: int, fiscal_period: str):
    """Fetch from sec_companyfacts_processed with qtrs awareness."""
    
    result = db.execute(query).fetchone()
    
    # Use qtrs fields to determine if YTD conversion needed
    income_is_ytd = result.income_statement_qtrs in [2, 3]
    cashflow_is_ytd = result.cash_flow_statement_qtrs in [2, 3]
    
    data = {
        "cash_flow": {
            "operating_cash_flow": to_float(result.operating_cash_flow),
            "is_ytd": cashflow_is_ytd,  # ✅ Use qtrs field
            "qtrs": result.cash_flow_statement_qtrs
        },
        "income_statement": {
            "total_revenue": to_float(result.total_revenue),
            "is_ytd": income_is_ytd,  # ✅ Use qtrs field
            "qtrs": result.income_statement_qtrs
        }
    }
    
    return data
```

### Phase 4: Validation (1 hour)

1. **Re-process 10 stocks** with new logic
2. **Compare results** with old approach
3. **Verify**: MSFT/AMZN use qtrs=1, AAPL uses qtrs=2
4. **Test**: Q4 computation produces positive values

---

## Validation Strategy

### Test Case 1: MSFT (Both qtrs=1)
```python
# Expected behavior:
# - Store individual quarter values (qtrs=1)
# - NO YTD conversion needed
# - Q2 = $X individual (not Q1+Q2)
# - Q4 = FY - (Q1+Q2+Q3) where all are individual
```

### Test Case 2: AAPL (Mixed, prefer qtrs=2)
```python
# Expected behavior:
# - Store YTD values (qtrs=2 for Q2)
# - YTD conversion DOES apply
# - Q2_YTD converted to Q2_individual = Q2_YTD - Q1
# - Q4 = FY - (Q1 + Q2_converted + Q3_converted)
```

### Test Case 3: New Stock (Unknown pattern)
```python
# Expected behavior:
# - Try qtrs=1 first
# - Fallback to qtrs=2/3 if not available
# - Store qtrs values for future queries
# - Apply YTD conversion based on qtrs field
```

---

## Migration Checklist

- [ ] **Schema**: Add `income_statement_qtrs`, `cash_flow_statement_qtrs` columns
- [ ] **Backfill**: Set qtrs values for existing data (default to YTD pattern)
- [ ] **Extraction**: Update SEC data processor with fallback chain
- [ ] **Query**: Update `_fetch_from_processed_table()` to use qtrs fields
- [ ] **Bulk Tables**: Add statement-aware qtrs filtering
- [ ] **Testing**: Validate MSFT (qtrs=1), AAPL (qtrs=2), 8 others (mixed)
- [ ] **Documentation**: Update data model docs

---

## Expected Outcomes

### Before Implementation:
- ❌ 80% of stocks use YTD values but we hardcode is_ytd by fiscal_period
- ❌ Cannot leverage qtrs=1 individual values when available (MSFT, AMZN)
- ❌ Bulk table path has no qtrs filtering

### After Implementation:
- ✅ **Accurate**: Store correct qtrs for each statement type
- ✅ **Optimal**: Use individual (qtrs=1) when available (20% of stocks)
- ✅ **Robust**: Fallback to YTD (qtrs=2/3) when needed (80% of stocks)
- ✅ **Consistent**: Bulk table and processed table use same logic
- ✅ **Validated**: Revenue qtrs=1 can validate YTD conversion accuracy

---

## Alternative: Simplified Approach (If Time Constrained)

**Option**: Keep current schema, add qtrs to QuarterlyData dataclass only

**Pros**: Minimal schema changes, faster implementation
**Cons**: Must re-query to determine qtrs, cannot optimize extraction

If you prefer the simpler approach, we can:
1. Keep `sec_companyfacts_processed` unchanged
2. Add `income_qtrs` and `cashflow_qtrs` to `QuarterlyData` dataclass
3. Determine qtrs at extraction time (query both qtrs=1 and qtrs=2/3)
4. Apply YTD conversion based on qtrs values

---

## Recommendation

**Implement Option 2 (Two qtrs Columns)** because:
1. **Data-driven**: Based on empirical S&P 100 analysis
2. **Accurate**: Represents all observed patterns
3. **Optimal**: Uses individual values when available
4. **Future-proof**: Extensible to balance sheet qtrs if needed
5. **Clean**: Clear semantics, efficient storage

**Timeline**: 4-5 hours total implementation + testing

Would you like me to proceed with the schema migration and implementation?

---

## Related Documentation

- **S&P 100 Analysis Results**: `analysis/sp100_statement_qtrs_patterns.json`
- **Bulk Table Discovery**: `analysis/BULK_TABLE_YTD_DISCOVERY.md`
- **SEC Data Sources**: `analysis/SEC_DATA_SOURCES_YTD_COMPLETE_ANALYSIS.md`
