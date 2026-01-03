# Valuation Table Logging Implementation Plan

## Overview

Replace verbose multi-line valuation logs with consolidated ASCII table format for better readability and comprehension.

**Status**: Table formatter utility created (`utils/valuation_table_formatter.py`)

## Current State (Verbose Logging)

### Example: DCF Valuation for UNH

```
2025-11-07 19:50:21,603 - utils.dcf_valuation - INFO - UNH - Using sector-based DCF parameters: Sector=Healthcare, Terminal Growth=3.5%, Projection Years=7
2025-11-07 19:50:21,609 - utils.dcf_valuation - INFO - 🔍 UNH - DCF FCF CALCULATION START
2025-11-07 19:50:47,641 - utils.dcf_valuation - INFO - 🔍 [DCF_INTEREST] UNH - Interest expense: $0.00B, Total debt: $76.90B, Cost of debt: 0.00%
2025-11-07 19:50:47,641 - utils.dcf_valuation - INFO - UNH - WACC inputs: Market Cap $301.52B, Debt $76.90B (weights E=0.80, D=0.20)
2025-11-07 19:50:47,641 - utils.dcf_valuation - INFO - UNH - WACC: 8.05% (raw: 8.05%, bounded 7-20%, tax rate 21%)
2025-11-07 19:50:47,642 - utils.dcf_valuation - INFO - UNH - Terminal Growth: 3.50% (base) +0.00% (Rule of 40: poor) = 3.50% (final)
2025-11-07 19:51:13,649 - utils.dcf_valuation - INFO - UNH - Discount Year 1: FCF $21.98B / (1+WACC)^1 = $20.34B
2025-11-07 19:51:13,649 - utils.dcf_valuation - INFO - UNH - Discount Year 2: FCF $22.71B / (1+WACC)^2 = $19.45B
... (5 more year lines)
2025-11-07 19:51:13,649 - utils.dcf_valuation - INFO - UNH - PV of FCF: $145.81B, PV of Terminal: $559.79B
2025-11-07 19:51:18,907 - utils.dcf_valuation - INFO - UNH - Fair Value: $676.02, Current: $324.21, Upside: +108.5%
```

**Issues**:
- Spans 15+ log lines
- Hard to see big picture
- Critical values scattered
- Difficult to compare across analyses

## Proposed State (Table Format)

### Example: DCF Valuation Table

```
====================================================================================================
  DCF VALUATION - UNH
====================================================================================================

────────────────────────────────────────────────────────────────────────────────────────────────────
  📊 INPUTS
────────────────────────────────────────────────────────────────────────────────────────────────────
  TTM Free Cash Flow         :   $18.24B
  TTM Revenue                :  $371.62B
  FCF Margin                 :      4.9%
  Historical FCF Growth      :     10.2%  (geometric mean)
  Revenue Growth             :      8.7%  (geometric mean)
  Rule of 40 Score           :     18.9%  (POOR)
  Projection Years           :        7

────────────────────────────────────────────────────────────────────────────────────────────────────
  💰 WACC CALCULATION
────────────────────────────────────────────────────────────────────────────────────────────────────
  Risk-Free Rate (10Y)       :     4.10%
  Beta                       :     0.68
  Equity Risk Premium        :     6.00%
  Cost of Equity             :     8.18%  (Rf + β × ERP)
  Cost of Debt (after-tax)   :     0.00%
  Market Cap                 :  $301.52B
  Total Debt                 :   $76.90B
  Equity Weight              :     79.7%
  Debt Weight                :     20.3%
  ═══════════════════════════════════════════════════════════
  WACC                       :     8.05%

────────────────────────────────────────────────────────────────────────────────────────────────────
  📈 FREE CASH FLOW PROJECTIONS & DISCOUNTING
────────────────────────────────────────────────────────────────────────────────────────────────────
  Year     Projected FCF    Discount Factor     Present Value
  -------- --------------- ------------------ ------------------
  Year 1       $21.98B           0.9255            $20.34B
  Year 2       $22.71B           0.8565            $19.45B
  Year 3       $24.23B           0.7927            $19.21B
  Year 4       $26.65B           0.7337            $19.55B
  Year 5       $30.20B           0.6788            $20.51B
  Year 6       $35.23B           0.6281            $22.15B
  Year 7       $42.28B           0.5818            $24.60B
  -------- --------------- ------------------ ------------------
  TOTAL                                           $145.81B

────────────────────────────────────────────────────────────────────────────────────────────────────
  🎯 TERMINAL VALUE
────────────────────────────────────────────────────────────────────────────────────────────────────
  Final Year FCF (Year 7)    :   $42.28B
  Terminal Growth Rate       :     3.50%
  Terminal FCF (perpetuity)  :   $43.76B
  Terminal Value             :  $962.34B  (FCF / (WACC - g))
  Discount Factor (Year 7)   :   0.5818
  Present Value (Terminal)   :  $559.79B

────────────────────────────────────────────────────────────────────────────────────────────────────
  💎 VALUATION SUMMARY
────────────────────────────────────────────────────────────────────────────────────────────────────
  PV of Projected FCF        :  $145.81B
  PV of Terminal Value       :  $559.79B
  ═══════════════════════════════════════════════════════════
  Enterprise Value           :  $705.60B
  Less: Net Debt             :   $76.90B
  Equity Value               :  $628.70B
  Shares Outstanding         :     0.93B
  ═══════════════════════════════════════════════════════════
  DCF Fair Value per Share   :  $676.02
  Current Price              :  $324.21
  Upside / (Downside)        :  +108.5%
====================================================================================================
```

**Benefits**:
- Single, comprehensive view
- Clear sections with visual hierarchy
- All critical data visible at once
- Easy to scan and understand
- Perfect for log reviews

## Implementation Plan

### Phase 1: DCF Valuation Table (PRIORITY)

**File**: `utils/dcf_valuation.py`

**Location**: After line 257 (current fair value log)

**Steps**:
1. Import `ValuationTableFormatter` from `utils.valuation_table_formatter`
2. Collect all intermediate calculation values into structured dicts
3. Call `ValuationTableFormatter.format_dcf_table()` with collected data
4. Log the formatted table using `logger.info(table_output)`

**Data to Collect**:
```python
# Inputs dict
inputs = {
    'ttm_fcf': ttm_fcf,
    'ttm_revenue': ttm_revenue,
    'fcf_margin': fcf_margin,
    'fcf_growth': fcf_growth_rate,
    'revenue_growth': revenue_growth_rate,
    'rule_of_40': rule_of_40_score,
    'rule_of_40_label': rule_of_40_classification,
    'projection_years': projection_years
}

# WACC breakdown dict
wacc_breakdown = {
    'rf_rate': risk_free_rate,
    'beta': beta,
    'erp': equity_risk_premium,
    'cost_of_equity': cost_of_equity,
    'cost_of_debt': cost_of_debt_after_tax,
    'market_cap': market_cap,
    'total_debt': total_debt,
    'equity_weight': equity_weight * 100,
    'debt_weight': debt_weight * 100,
    'wacc': wacc
}

# Projections list
projections = [
    {
        'year': 1,
        'fcf': fcf_year_1,
        'discount_factor': 1 / (1 + wacc) ** 1,
        'pv_fcf': pv_fcf_year_1
    },
    # ... repeat for all projection years
]

# Terminal dict
terminal = {
    'terminal_growth': terminal_growth_rate,
    'terminal_value': terminal_value,
    'discount_factor': 1 / (1 + wacc) ** projection_years,
    'pv_terminal': pv_terminal_value
}

# Valuation dict
valuation = {
    'pv_fcf': sum_pv_fcf,
    'pv_terminal': pv_terminal_value,
    'enterprise_value': enterprise_value,
    'net_debt': net_debt,
    'equity_value': equity_value,
    'shares_outstanding': shares_outstanding,
    'fair_value': fair_value_per_share,
    'current_price': current_price,
    'upside_pct': upside_downside
}
```

**Integration**:
```python
# After calculating final_fair_value (around line 257)
from utils.valuation_table_formatter import ValuationTableFormatter

# Format and log DCF table
dcf_table = ValuationTableFormatter.format_dcf_table(
    symbol=self.symbol,
    inputs=inputs,
    wacc_breakdown=wacc_breakdown,
    projections=projections,
    terminal=terminal,
    valuation=valuation
)
logger.info(dcf_table)
```

### Phase 2: Relative Valuation Table

**File**: `src/investigator/domain/agents/fundamental.py`

**Location**: After all P/E, P/S, P/B, EV/EBITDA calculations (in `_perform_valuation()`)

**Steps**:
1. Collect all relative valuation results
2. Call `ValuationTableFormatter.format_relative_valuation_table()`
3. Log consolidated table

**Example Table Output**:
```
========================================================================================================================
  RELATIVE VALUATION - PLTR  (Current Price: $177.93)
========================================================================================================================

  Model        Metric               Value        Sector Multiple    Fair Value     vs Current    Conf
  ------------ -------------------- ------------ ------------------ --------------- ------------ ------
  P/E          Net Income           $1.10B                  28.5x       $93.93         -47.2%     80%
              └─ Sector: Technology (Software) Median P/E

  EV/EBITDA    EBITDA               $1.50B                  22.0x       $99.84         -43.9%     75%
              └─ Sector: Technology (Software) Median EV/EBITDA

  P/S          Revenue              $2.85B                   9.2x       $95.35         -46.4%     60%
              └─ Sector: Technology (Software) Median P/S

  P/B          Book Value           $5.20B                   7.5x      $153.31         -13.8%     50%
              └─ Sector: Technology (Software) Median P/B
========================================================================================================================
```

### Phase 3: GGM Table

**File**: `utils/gordon_growth_model.py`

**Location**: In `calculate_ggm()` method after fair value calculation

**Steps**:
1. Collect historical dividends (last 12 quarters)
2. Collect GGM inputs (dividend, payout ratio, growth rate, cost of equity)
3. Optionally collect multi-stage projections if used
4. Call `ValuationTableFormatter.format_ggm_table()`
5. Log table

**Example Table Output**:
```
====================================================================================================
  GORDON GROWTH MODEL (GGM) - JNJ
====================================================================================================

────────────────────────────────────────────────────────────────────────────────────────────────────
  📊 HISTORICAL DIVIDENDS (Last 12 Quarters)
────────────────────────────────────────────────────────────────────────────────────────────────────
  Quarter         Dividend per Share
  --------------- --------------------
  2022-Q1                    $1.0600
  2022-Q2                    $1.1300
  2022-Q3                    $1.1300
  2022-Q4                    $1.1300
  2023-Q1                    $1.1300
  2023-Q2                    $1.1900
  2023-Q3                    $1.1900
  2023-Q4                    $1.1900
  2024-Q1                    $1.1900
  2024-Q2                    $1.2400
  2024-Q3                    $1.2400
  2024-Q4 (est)              $1.2400

────────────────────────────────────────────────────────────────────────────────────────────────────
  💰 GGM INPUTS
────────────────────────────────────────────────────────────────────────────────────────────────────
  Current Dividend (annual)  :   $4.8400
  Payout Ratio               :     52.3%
  Dividend Growth Rate       :      4.20%  (historical average)
  Cost of Equity             :      7.85%

────────────────────────────────────────────────────────────────────────────────────────────────────
  💎 GGM VALUATION
────────────────────────────────────────────────────────────────────────────────────────────────────
  Formula: V = D₁ / (r - g)
           D₁ = $5.0437  (next year dividend)
           r  = 7.85%  (cost of equity)
           g  = 4.20%  (growth rate)
           r - g = 3.65%
  ═══════════════════════════════════════════════════════════
  GGM Fair Value per Share   :  $138.15
  Current Price              :  $148.50
  Upside / (Downside)        :   -6.9%
====================================================================================================
```

### Phase 4: Comprehensive Valuation Summary Table

**File**: `src/investigator/domain/agents/fundamental.py`

**Location**: After all models calculated, before blending (in `_perform_valuation()`)

**Steps**:
1. Collect all model results with fair values, confidence scores, weights
2. Call `ValuationTableFormatter.format_valuation_summary_table()`
3. Log comprehensive summary

**Example Table Output**:
```
====================================================================================================
  VALUATION SUMMARY - PLTR
====================================================================================================

  Model           Fair Value     Confidence      Weight     Weighted FV
  --------------- --------------- ------------ ---------- ---------------
  DCF                  $24.02         65%           0%             -
  P/E                  $93.93         80%         100%        $93.93
  EV/EBITDA            $99.84         75%           0%             -
  P/S                  $95.35         60%           0%             -
  P/B                 $153.31         50%           0%             -
  GGM                       -           -            0%             -
  --------------- --------------- ------------ ---------- ---------------
  BLENDED                                        100%        $93.93

────────────────────────────────────────────────────────────────────────────────────────────────────
  Tier Classification        : balanced_default
  Blended Fair Value         :   $93.93
  Current Price              :  $177.93
  Upside / (Downside)        :   -47.2%
====================================================================================================
```

## Testing Strategy

### Unit Test

Create `tests/unit/utils/test_valuation_table_formatter.py`:

```python
import pytest
from utils.valuation_table_formatter import ValuationTableFormatter


def test_format_dcf_table():
    """Test DCF table formatting with sample data."""
    inputs = {
        'ttm_fcf': 18.24e9,
        'ttm_revenue': 371.62e9,
        'fcf_margin': 4.9,
        'fcf_growth': 10.2,
        'revenue_growth': 8.7,
        'rule_of_40': 18.9,
        'rule_of_40_label': 'POOR',
        'projection_years': 7
    }

    wacc_breakdown = {
        'rf_rate': 4.10,
        'beta': 0.68,
        'erp': 6.00,
        'cost_of_equity': 8.18,
        'cost_of_debt': 0.00,
        'market_cap': 301.52e9,
        'total_debt': 76.90e9,
        'equity_weight': 79.7,
        'debt_weight': 20.3,
        'wacc': 8.05
    }

    projections = [
        {'year': 1, 'fcf': 21.98e9, 'discount_factor': 0.9255, 'pv_fcf': 20.34e9},
        {'year': 2, 'fcf': 22.71e9, 'discount_factor': 0.8565, 'pv_fcf': 19.45e9},
        # ... etc
    ]

    terminal = {
        'terminal_growth': 3.50,
        'terminal_value': 962.34e9,
        'discount_factor': 0.5818,
        'pv_terminal': 559.79e9
    }

    valuation = {
        'pv_fcf': 145.81e9,
        'pv_terminal': 559.79e9,
        'enterprise_value': 705.60e9,
        'net_debt': 76.90e9,
        'equity_value': 628.70e9,
        'shares_outstanding': 0.93e9,
        'fair_value': 676.02,
        'current_price': 324.21,
        'upside_pct': 108.5
    }

    table = ValuationTableFormatter.format_dcf_table(
        symbol='UNH',
        inputs=inputs,
        wacc_breakdown=wacc_breakdown,
        projections=projections,
        terminal=terminal,
        valuation=valuation
    )

    # Assertions
    assert 'DCF VALUATION - UNH' in table
    assert '$18.24B' in table  # TTM FCF
    assert '8.05%' in table  # WACC
    assert '$676.02' in table  # Fair value
    assert '+108.5%' in table  # Upside
```

### Integration Test

Run UNH analysis and verify table appears in logs:

```bash
python3 cli_orchestrator.py analyze UNH -m standard --force-refresh -o /tmp/unh_table_test.json 2>&1 | tee /tmp/unh_table_test.log

# Check for table output
grep -A 50 "DCF VALUATION - UNH" /tmp/unh_table_test.log
```

## Rollout Plan

1. **Phase 1 (DCF)**: Implement table formatter in `utils/dcf_valuation.py`
2. **Phase 2 (Relative)**: Add to `fundamental.py` for P/E, P/S, P/B, EV/EBITDA
3. **Phase 3 (GGM)**: Add to `utils/gordon_growth_model.py`
4. **Phase 4 (Summary)**: Add comprehensive summary table to `fundamental.py`
5. **Cleanup**: Remove verbose individual log lines (keep only table output)

## Benefits

- **Readability**: Single-glance comprehension of entire valuation
- **Debugging**: Easier to spot calculation errors
- **Comparison**: Simpler to compare valuations across different symbols
- **Documentation**: Logs serve as self-documenting valuation reports
- **Professionalism**: Clean, structured output befitting institutional-grade analysis

## Next Steps

1. ✅ Created table formatter utility (`utils/valuation_table_formatter.py`)
2. ⏳ Integrate DCF table into `utils/dcf_valuation.py`
3. ⏳ Integrate relative valuation table into `fundamental.py`
4. ⏳ Integrate GGM table into `utils/gordon_growth_model.py`
5. ⏳ Add comprehensive summary table
6. ⏳ Create unit tests
7. ⏳ Run integration tests
8. ⏳ Remove verbose logging once tables verified
