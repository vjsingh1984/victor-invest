# /tmp/aapl.log Analysis - Critical Issues Found

**Date**: 2025-11-04
**Log File**: /tmp/aapl.log (1390 lines)
**Symbol**: AAPL
**Status**: 🔴 **DCF STILL FAILING** despite architectural improvements

---

## CRITICAL ISSUE #1: Statement-Level Structure Breaking `_fetch_historical_quarters()`

### Root Cause
**File**: `src/investigator/domain/agents/fundamental.py:770`

The `_fetch_historical_quarters()` method is trying to access a key `'financial_data'` that no longer exists after implementing statement-level architecture.

### Error from Log
```
Line 312: 2025-11-04 13:53:03,670 - agent.fund_agent_1 - WARNING - Multi-quarter analysis failed for AAPL: Failed to fetch historical quarters for AAPL: 'financial_data'
```

### Flow of the Problem
1. `_fetch_from_processed_table()` (line 1076-1166) returns **statement-level structure** (new):
   ```python
   {
       "income_statement": {..., "is_ytd": True},
       "cash_flow": {..., "is_ytd": True},
       "balance_sheet": {...},
       "ratios": {...}
   }
   ```

2. `_fetch_historical_quarters()` (line 770) expects **flat structure with financial_data** (old):
   ```python
   {
       "financial_data": {
           "operating_cash_flow": 123,
           "capital_expenditures": 456
       }
   }
   ```

3. KeyError `'financial_data'` is raised
4. Exception handler (line 488-491) catches it and sets:
   ```python
   company_data["quarterly_data"] = []  # EMPTY LIST!
   ```

5. DCF/GGM receive empty `quarterly_data` (line 2562)
6. DCF calculates $0.0M FCF (line 669)

### Fix Required
**Update `_fetch_historical_quarters()` to work with statement-level structure returned by `_fetch_from_processed_table()`**

The method needs to:
- Accept data with `cash_flow`, `income_statement`, `balance_sheet` keys
- Extract metrics from statement-level structure
- Convert to `QuarterlyData` objects with flattened structure (or update `QuarterlyData` class)

---

## CRITICAL ISSUE #2: Duplicate Market Data Queries (60 queries!)

### Problem
The technical analysis agent is fetching market data **60 times** for different time periods and different ETFs:

```
Fetching 10 days for: SPY, QQQ, IWM, EFA, EEM, AGG, TLT, HYG, GLD, SLV, USO, DBC, GSG (13 symbols)
Fetching 21 days for: Same 13 symbols
Fetching 63 days for: Same 13 symbols
Fetching 253 days for: Same 13 symbols

Total = 13 symbols × 4 time periods = 52 queries for ETFs
      + 1 symbol (AAPL) × 4 time periods = 4 queries
      + misc = ~60 queries
```

### Recommendation (from user)
**Fetch 252 days ONCE, then slice the array:**

```python
# Instead of this (current):
data_10d = fetch_market_data(symbol, 10)
data_21d = fetch_market_data(symbol, 21)
data_63d = fetch_market_data(symbol, 63)
data_253d = fetch_market_data(symbol, 253)

# Do this (optimized):
data_252d = fetch_market_data(symbol, 252)  # ONE query
data_10d = data_252d[-10:]  # Slice
data_21d = data_252d[-21:]  # Slice
data_63d = data_252d[-63:]  # Slice
```

### Impact
- **Performance**: 60 DB queries → 14 queries (1 for AAPL + 13 for ETFs) = **77% reduction**
- **Time Savings**: ~3-5 seconds per analysis

### File to Modify
`agents/technical_agent.py` or `utils/market_data_fetcher.py`

---

## CRITICAL ISSUE #3: Missing Quarterly Analysis Logging

### Problem
No logging for quarterly analysis steps, making it hard to track:
- Which quarters are being fetched
- How many quarters are available
- Which analysis methods are being run

### From Log
Only see:
```
Line 309: 🔍 [PROCESSED_TABLE] Querying for AAPL 2018-FY...
Line 310: ✅ [PROCESSED_TABLE] Found data: Revenue=$215.64B...
Line 312: WARNING - Multi-quarter analysis failed...  ← NO DETAIL!
```

### Recommendation (from user)
Add comprehensive logging in fundamental analysis:

```python
# In _fetch_historical_quarters():
self.logger.info(f"📊 [FETCH_QUARTERS] Fetching {num_quarters} quarters for {symbol}")
self.logger.info(f"📊 [FETCH_QUARTERS] Retrieved {len(quarters)} quarters from database")
for q in quarters:
    self.logger.debug(f"  Q: {q['fiscal_year']}-{q['fiscal_period']} OCF=${q['cash_flow']['operating_cash_flow']/1e9:.1f}B")

# In SEC analysis processing:
self.logger.info(f"🔍 [SEC_PROCESS] Starting quarterly metrics extraction for {symbol}")
self.logger.info(f"🔍 [SEC_PROCESS] Extracted {len(quarterly_data)} quarters")
```

### Impact
- Better debugging
- Easier to identify data flow issues
- Can track which quarters have YTD vs individual values

---

## ISSUE #4: YTD Conversion Not Yet Tested

### Status
- ✅ YTD conversion logic implemented in `quarterly_calculator.py:202-320`
- ❓ **NOT TESTED** because `_fetch_historical_quarters()` is failing before YTD conversion runs

### Next Step
After fixing Issue #1, verify YTD conversion actually works:
- Check that Q2/Q3 have `is_ytd=True` in database
- Verify conversion subtracts Q1 from Q2, etc.
- Ensure DCF receives individual quarter values, not YTD cumulative

---

## Summary of Failures

| Issue | Status | Impact | Priority |
|-------|--------|--------|----------|
| Statement-level structure breaking _fetch_historical_quarters | 🔴 **BLOCKING** | DCF returns $0 | **P0** |
| 60 duplicate market data queries | 🟡 Performance | Slow analysis | **P1** |
| Missing quarterly analysis logging | 🟡 Debugging | Hard to troubleshoot | **P1** |
| YTD conversion untested | 🟡 Unknown | May still be broken | **P2** |

---

## Recommended Fix Order

### 1. **URGENT: Fix `_fetch_historical_quarters()`** (P0)
**File**: `src/investigator/domain/agents/fundamental.py:770`

Update to work with statement-level structure from `_fetch_from_processed_table()`:

```python
async def _fetch_historical_quarters(self, symbol: str, num_quarters: int = 8) -> List[QuarterlyData]:
    """
    Fetch historical quarterly data using statement-level structure

    NEW: Works with cash_flow, income_statement, balance_sheet structure
    """
    try:
        # Fetch from processed table (returns statement-level structure)
        quarters = self._fetch_from_processed_table(symbol, max_periods=num_quarters)

        self.logger.info(f"📊 [FETCH_QUARTERS] Retrieved {len(quarters)} quarters for {symbol}")

        # Convert statement-level structure to QuarterlyData objects
        quarterly_data_list = []
        for q in quarters:
            # Extract from statement-level structure
            cash_flow = q.get('cash_flow', {})
            income_stmt = q.get('income_statement', {})
            balance_sheet = q.get('balance_sheet', {})

            # Create flattened dict for QuarterlyData
            data = {
                'fiscal_year': q['fiscal_year'],
                'fiscal_period': q['fiscal_period'],
                'operating_cash_flow': cash_flow.get('operating_cash_flow', 0),
                'capital_expenditures': cash_flow.get('capital_expenditures', 0),
                'free_cash_flow': cash_flow.get('free_cash_flow', 0),
                'dividends_paid': cash_flow.get('dividends_paid', 0),
                'total_revenue': income_stmt.get('total_revenue', 0),
                'net_income': income_stmt.get('net_income', 0),
                # ... more fields
            }

            quarterly_data_list.append(QuarterlyData(**data))

        return quarterly_data_list

    except Exception as e:
        self.logger.error(f"Failed to fetch historical quarters for {symbol}: {e}", exc_info=True)
        raise  # Don't swallow the exception with empty list!
```

### 2. **Optimize Market Data Fetching** (P1)
**File**: `utils/market_data_fetcher.py` or where market data is fetched

Add method to fetch once and slice:

```python
def fetch_market_data_multi_period(self, symbol: str, periods: List[int] = [10, 21, 63, 253]):
    """
    Fetch market data once and slice for multiple periods

    Returns dict with keys for each period: {10: data_10d, 21: data_21d, ...}
    """
    max_period = max(periods)
    full_data = self.fetch_market_data(symbol, days=max_period)

    result = {}
    for period in periods:
        result[period] = full_data[-period:] if len(full_data) >= period else full_data

    return result
```

### 3. **Add Comprehensive Logging** (P1)
**Files**: `fundamental.py`, `sec_companyfacts_extractor.py`

Add detailed logging at each step of quarterly data flow:
- When fetching from database
- When converting YTD
- When passing to DCF/GGM
- Include quarter count, date ranges, sample values

---

## Test Plan After Fix

```bash
# 1. Clear all caches
rm -rf data/llm_cache/AAPL data/sec_cache/facts/facts/AAPL
PGPASSWORD=investigator psql -h ${DB_HOST:-localhost} -U investigator -d sec_database \
  -c "DELETE FROM llm_responses WHERE symbol = 'AAPL';"

# 2. Run analysis with verbose logging
python3 cli_orchestrator.py analyze AAPL -m standard --force-refresh 2>&1 | tee /tmp/aapl_fix_test.log

# 3. Verify success
grep -E "FETCH_QUARTERS|quarterly_data|TTM FCF|Fair Value" /tmp/aapl_fix_test.log

# Expected output:
# ✅ [FETCH_QUARTERS] Retrieved 8 quarters for AAPL
# ✅ Added 8 quarters to company_data for AAPL
# ✅ YTD to quarterly conversion complete for 2 fiscal years
# ✅ AAPL - Latest TTM FCF: $86000.0M (should be > 0!)
# ✅ AAPL - DCF Fair Value: $215.50 (should be > 0!)
```

---

## Conclusion

The DCF is still failing because the statement-level architecture refactor broke the `_fetch_historical_quarters()` method, which expects the old `financial_data` flat structure. This causes a KeyError, which is caught and results in empty `quarterly_data`, leading to $0.0M FCF.

**Fix Priority: P0 - URGENT**
**Estimated Fix Time**: 15-30 minutes
**Risk**: Low (just need to update one method to read from new structure)
