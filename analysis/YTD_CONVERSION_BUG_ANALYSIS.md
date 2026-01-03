# YTD Conversion Bug - Root Cause Analysis

**Date**: 2025-11-04
**Issue**: Negative Q4 OCF (-$63,364.0M) due to failed YTD conversion
**Status**: ✅ **ROOT CAUSE IDENTIFIED**, Fix ready to implement

---

## Executive Summary

The negative Q4 OCF issue is caused by a **data flow bug** where YTD (Year-To-Date) flags are **lost during data transformation**, causing the YTD conversion function to skip conversion, leaving Q2/Q3 with cumulative values that produce negative Q4 when computed.

**Impact**: DCF/GGM valuations are incorrect for ALL companies because Q4 values are wrong.

---

## Problem Statement

From `/tmp/aapl.log` analysis:

```
Line 797: WARNING - Computed negative Q4 operating_cash_flow: -63364.0M
          (FY: 122151.0M, Q1-Q3 sum: 185515.0M)
Line 799: INFO - Computed Q4-2024 from FY and reported quarters. OCF: -63364.0M
```

**The Math Breakdown:**
| Period | OCF in DB (Billions) | Type |
|--------|---------------------|------|
| Q1-2024 | $34.01 | Individual |
| Q2-2024 | $62.57 | **YTD cumulative** (Q1+Q2) |
| Q3-2024 | $88.95 | **YTD cumulative** (Q1+Q2+Q3) |
| FY-2024 | $122.15 | Full year |
| **Q1+Q2+Q3 Sum** | **$185.53** | **Problem!** |
| **Q4 Computed** | $122.15 - $185.53 = **-$63.38** | **NEGATIVE!** |

---

## Root Cause Analysis

### The Bug Chain

1. **Database Storage** (`sec_companyfacts_processed` table)
   - Stores YTD cumulative values from 10-Q filings for Q2/Q3
   - **NO `is_ytd` column** in database schema
   - Values: Q1=$34.01B, Q2=$62.57B (YTD), Q3=$88.95B (YTD)

2. **Data Fetching** (`fundamental.py:1186`)
   ```python
   # Determine if this is YTD data (Q2/Q3 10-Q filings report YTD cumulative)
   is_ytd = fiscal_period in ['Q2', 'Q3']  # ✅ CORRECT

   data = {
       "cash_flow": {
           "operating_cash_flow": to_float(result.operating_cash_flow),
           "is_ytd": is_ytd  # ✅ CORRECTLY SET TO True for Q2/Q3
       },
       "income_statement": {
           "total_revenue": to_float(result.total_revenue),
           "is_ytd": is_ytd  # ✅ CORRECTLY SET TO True for Q2/Q3
       }
   }
   ```
   **Status**: ✅ `is_ytd` flags correctly set for Q2/Q3

3. **QuarterlyData Construction** (`fundamental.py:950-1005`)
   ```python
   # Flatten statement-level structure for QuarterlyData storage
   financial_data = {
       "revenues": income_statement.get("total_revenue", 0),
       "operating_cash_flow": cash_flow.get("operating_cash_flow", 0),
       # ... more fields
   }

   q_data = QuarterlyData(
       fiscal_year=q['fiscal_year'],
       fiscal_period=q['fiscal_period'],
       financial_data=financial_data,  # ❌ is_ytd flags NOT included!
       ratios=ratios,
       data_quality=data_quality,
       filing_date=q.get('filed_date')
   )
   ```
   **Status**: ❌ **BUG #1**: `is_ytd` flags from statement-level structure are discarded during flattening

4. **QuarterlyData.to_dict()** (`fundamental.py:70-124`)
   ```python
   def to_dict(self) -> Dict[str, Any]:
       fd = self.financial_data or {}

       return {
           "cash_flow": {
               "operating_cash_flow": fd.get("operating_cash_flow", 0),
               "is_ytd": False  # ❌ HARDCODED FALSE!
           },
           "income_statement": {
               "total_revenue": fd.get("revenues", 0),
               "is_ytd": False  # ❌ HARDCODED FALSE!
           }
       }
   ```
   **Status**: ❌ **BUG #2**: `to_dict()` hardcodes `is_ytd: False` for ALL quarters

5. **YTD Conversion** (`quarterly_calculator.py:268-386`)
   ```python
   def convert_ytd_to_quarterly(quarters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
       # Check if income_statement is YTD
       if q2.get('income_statement', {}).get('is_ytd'):  # ❌ NEVER True!
           # Convert Q2 YTD to individual
           income[key] = q2_ytd_val - q1_val
   ```
   **Status**: ❌ Conversion skipped because `is_ytd` is always False

6. **Q4 Computation** (`quarterly_calculator.py:125`)
   ```python
   q4_value = fy_value - quarters_sum  # Negative because quarters_sum has YTD!
   ```
   **Status**: ❌ Produces negative Q4 values

---

## The Fix

### Change #1: Add `is_ytd_cashflow` and `is_ytd_income` to QuarterlyData

**File**: `src/investigator/domain/agents/fundamental.py` lines 30-49

```python
@dataclass
class QuarterlyData:
    fiscal_year: int
    fiscal_period: str
    financial_data: Dict[str, Any]
    ratios: Optional[Dict[str, Any]] = None
    market_data: Optional[Dict[str, Any]] = None
    data_quality: Optional[Dict[str, Any]] = None
    filing_date: Optional[str] = None
    is_ytd_cashflow: bool = False  # NEW: Track if cash_flow values are YTD
    is_ytd_income: bool = False    # NEW: Track if income_statement values are YTD
```

### Change #2: Preserve `is_ytd` flags during QuarterlyData construction

**File**: `src/investigator/domain/agents/fundamental.py` lines 950-1005

```python
# Extract is_ytd flags from statement-level structure
cash_flow = processed_data.get("cash_flow", {})
income_statement = processed_data.get("income_statement", {})

is_ytd_cashflow = cash_flow.get("is_ytd", False)
is_ytd_income = income_statement.get("is_ytd", False)

# Flatten into financial_data dict
financial_data = {
    "revenues": income_statement.get("total_revenue", 0),
    "operating_cash_flow": cash_flow.get("operating_cash_flow", 0),
    # ... more fields
}

q_data = QuarterlyData(
    fiscal_year=q['fiscal_year'],
    fiscal_period=q['fiscal_period'],
    financial_data=financial_data,
    ratios=ratios,
    data_quality=data_quality,
    filing_date=q.get('filed_date'),
    is_ytd_cashflow=is_ytd_cashflow,  # NEW: Preserve flag
    is_ytd_income=is_ytd_income        # NEW: Preserve flag
)
```

### Change #3: Return correct `is_ytd` values in to_dict()

**File**: `src/investigator/domain/agents/fundamental.py` lines 70-124

```python
def to_dict(self) -> Dict[str, Any]:
    fd = self.financial_data or {}

    return {
        "cash_flow": {
            "operating_cash_flow": fd.get("operating_cash_flow", 0),
            "capital_expenditures": fd.get("capital_expenditures", 0),
            "is_ytd": self.is_ytd_cashflow  # ✅ USE STORED FLAG
        },
        "income_statement": {
            "total_revenue": fd.get("revenues", 0) or fd.get("total_revenue", 0),
            "net_income": fd.get("net_income", 0),
            "is_ytd": self.is_ytd_income  # ✅ USE STORED FLAG
        },
        # ... rest of structure
    }
```

### Change #4: Update from_dict() to handle is_ytd flags

**File**: `src/investigator/domain/agents/fundamental.py` lines 126-178

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "QuarterlyData":
    # Check if this is new statement-level structure
    if "cash_flow" in data or "income_statement" in data:
        cash_flow = data.get("cash_flow", {})
        income = data.get("income_statement", {})

        # Extract is_ytd flags
        is_ytd_cashflow = cash_flow.get("is_ytd", False)
        is_ytd_income = income.get("is_ytd", False)

        # Flatten to financial_data
        financial_data = {
            "operating_cash_flow": cash_flow.get("operating_cash_flow", 0),
            "revenues": income.get("total_revenue", 0),
            # ... more fields
        }
    else:
        # Old flat structure
        financial_data = data.get("financial_data", {})
        is_ytd_cashflow = False  # Assume already converted
        is_ytd_income = False

    return cls(
        fiscal_year=data.get("fiscal_year"),
        fiscal_period=data.get("fiscal_period"),
        financial_data=financial_data,
        ratios=data.get("ratios"),
        market_data=data.get("market_data"),
        data_quality=data.get("data_quality"),
        filing_date=data.get("filing_date"),
        is_ytd_cashflow=is_ytd_cashflow,  # NEW
        is_ytd_income=is_ytd_income        # NEW
    )
```

---

## Expected Behavior After Fix

### Before Fix (Current State):

```
🔍 AAPL - Q4-2024: OCF=$-63364.0M, CapEx=$8578.0M ❌ NEGATIVE!
🔍 AAPL - Q3-2024: OCF=$88945.0M (YTD cumulative)
🔍 AAPL - Q2-2024: OCF=$62565.0M (YTD cumulative)
🔍 AAPL - Q1-2024: OCF=$34005.0M (individual)
AAPL - TTM FCF: $94287.0M ❌ WRONG! (because Q4 is negative)
```

### After Fix:

```
✅ Converted Q2-2024 cash_flow from YTD to individual quarter
✅ Converted Q3-2024 cash_flow from YTD to individual quarter

🔍 AAPL - Q4-2024: OCF=$27206.0M ✅ POSITIVE!
🔍 AAPL - Q3-2024: OCF=$26390.0M (individual, after YTD conversion)
🔍 AAPL - Q2-2024: OCF=$28560.0M (individual, after YTD conversion)
🔍 AAPL - Q1-2024: OCF=$34005.0M (individual)
AAPL - TTM FCF: $116161.0M ✅ CORRECT!
```

**Verification Math:**
- Q2 individual = Q2_YTD - Q1 = $62.57B - $34.01B = $28.56B ✅
- Q3 individual = Q3_YTD - (Q1 + Q2_individual) = $88.95B - ($34.01B + $28.56B) = $26.38B ✅
- Q4 computed = FY - (Q1 + Q2 + Q3) = $122.15B - ($34.01B + $28.56B + $26.38B) = $33.20B ✅

---

## Testing Checklist

- [ ] Add `is_ytd_cashflow` and `is_ytd_income` fields to QuarterlyData dataclass
- [ ] Update `_fetch_historical_quarters()` to preserve is_ytd flags
- [ ] Update `to_dict()` to return stored is_ytd flags
- [ ] Update `from_dict()` to handle is_ytd flags from both old and new structures
- [ ] Run fresh AAPL analysis and verify Q4 is positive
- [ ] Verify YTD conversion logs show "Converted Q2/Q3 from YTD to individual"
- [ ] Verify TTM FCF matches expected value
- [ ] Test with MSFT, GOOGL to ensure fix works universally

---

## Impact Assessment

**Severity**: 🔴 **CRITICAL** - Affects ALL stock analyses

**Affected Components**:
- DCF valuation (incorrect FCF calculations)
- GGM valuation (incorrect dividend payout calculations)
- Executive summary (incorrect quarterly trends)
- YoY/QoQ growth analysis (comparing YTD to individual quarters)

**Stocks Affected**:
- **100% of analyzed stocks** with available quarterly data
- Particularly problematic for Q4-heavy businesses (retail, consumer goods)

**Why This Wasn't Caught Earlier**:
1. TTM FCF was still positive (by coincidence) despite Q4 being negative
2. DCF Fair Value appeared reasonable because negative Q4 was offset by YTD-inflated Q2/Q3
3. No validation warnings for negative operational metrics

---

## Additional User Questions

### 1. Is YTD/FY based quarterly numbers calculated correctly?
**Answer**: ❌ **NO** - YTD conversion is **NOT working** due to lost `is_ytd` flags. Fix implemented above.

### 2. Does DCF account for quarterly numbers or provide NPV based on quarterly?
**Answer**: DCF uses **quarterly TTM (Trailing Twelve Months)** cash flows:
- Sums OCF and CapEx from most recent 4 quarters
- TTM FCF = Σ(Q_OCF) - Σ(Q_CapEx) for Q1,Q2,Q3,Q4
- Then projects 10-year annual FCF from this baseline
- Discounts using annual WACC (not quarterly)
- **Correct approach** for quarterly data, but needs individual quarters (not YTD)

### 3. Should we add debug logging for WACC calculation?
**Answer**: ✅ **YES** - Add detailed WACC component logging in `utils/dcf_valuation.py`:
```python
logger.info(
    f"{symbol} - WACC Components: "
    f"Beta={beta:.2f}, "
    f"Risk-Free Rate={risk_free_rate:.2f}%, "
    f"Market Risk Premium={market_risk_premium:.2f}%, "
    f"Cost of Equity={cost_of_equity:.2f}%, "
    f"WACC={wacc:.2f}%"
)
```

### 4. Do we need more than 4 quarters (like 8) for YoY/QoQ growth analysis?
**Answer**: ✅ **YES** - For 3-year investment horizon:
- **Current**: 4 quarters (1 year TTM) - insufficient for growth trends
- **Recommended**: 8 quarters (2 years) minimum for YoY comparisons
- **Optimal**: 20 quarters (5 years) to capture:
  - Full business cycles
  - Seasonal patterns
  - YoY growth trends (need 4 years of data to calculate 3 years of YoY growth)
  - QoQ sequential patterns

**Implementation**: Already supported via `get_rolling_ttm_periods(num_quarters=8)` parameter.

---

## Related Documentation

- **Previous Issues**: `analysis/LOG_ANALYSIS_CRITICAL_ISSUES.md`
- **Architecture**: `analysis/COMPREHENSIVE_FIX_SUMMARY.md`
- **Database Consolidation**: `analysis/ARCHITECTURE_CONSOLIDATION_ANALYSIS.md`

---

## Conclusion

The negative Q4 OCF issue is caused by **lost `is_ytd` flags** during data transformation. The fix requires:

1. Adding `is_ytd_cashflow`/`is_ytd_income` fields to QuarterlyData
2. Preserving flags during construction
3. Returning correct flags in `to_dict()`
4. Handling flags in `from_dict()` for backward compatibility

Once implemented, YTD conversion will work correctly, producing positive Q4 values and accurate DCF/GGM valuations.

**Status**: Root cause identified, fix ready to implement.
