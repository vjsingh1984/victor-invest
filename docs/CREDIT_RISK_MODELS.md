# Credit Risk Models

This document describes the credit risk models implemented in Victor-Invest for assessing company financial health and bankruptcy risk.

## Table of Contents

1. [Overview](#overview)
2. [Altman Z-Score](#altman-z-score)
3. [Beneish M-Score](#beneish-m-score)
4. [Piotroski F-Score](#piotroski-f-score)
5. [Composite Distress Tier](#composite-distress-tier)
6. [CLI Usage](#cli-usage)

---

## Overview

Victor-Invest implements three complementary credit risk models:

| Model | Purpose | Score Range | Best For |
|-------|---------|-------------|----------|
| **Altman Z-Score** | Bankruptcy prediction | -∞ to +∞ | Financial distress |
| **Beneish M-Score** | Earnings manipulation | -∞ to +∞ | Fraud detection |
| **Piotroski F-Score** | Financial strength | 0-9 | Value investing |

These models use publicly available SEC financial data (10-K, 10-Q) and require no proprietary data sources.

---

## Altman Z-Score

### Purpose
Predicts the probability of bankruptcy within 2 years.

### Formula (Manufacturing Companies)

```
Z = 1.2×A + 1.4×B + 3.3×C + 0.6×D + 1.0×E

Where:
A = Working Capital / Total Assets
B = Retained Earnings / Total Assets
C = EBIT / Total Assets
D = Market Value of Equity / Total Liabilities
E = Sales / Total Assets
```

### Formula (Non-Manufacturing Companies)

```
Z = 6.56×A + 3.26×B + 6.72×C + 1.05×D

Where:
A = Working Capital / Total Assets
B = Retained Earnings / Total Assets
C = EBIT / Total Assets
D = Book Value of Equity / Total Liabilities
```

### Interpretation

| Z-Score Range | Zone | Interpretation |
|--------------|------|----------------|
| > 2.99 | Safe | Low bankruptcy risk |
| 1.81 - 2.99 | Grey | Moderate uncertainty |
| < 1.81 | Distress | High bankruptcy risk |

### Limitations

- Designed for manufacturing companies (1960s)
- Less accurate for service/tech companies
- Does not account for off-balance-sheet items
- May miss rapid deterioration

---

## Beneish M-Score

### Purpose
Detects earnings manipulation and accounting fraud.

### Formula

```
M = -4.84 + 0.920×DSRI + 0.528×GMI + 0.404×AQI
    + 0.892×SGI + 0.115×DEPI - 0.172×SGAI
    + 4.679×TATA - 0.327×LVGI

Where:
DSRI = Days Sales Receivables Index
GMI  = Gross Margin Index
AQI  = Asset Quality Index
SGI  = Sales Growth Index
DEPI = Depreciation Index
SGAI = SG&A Expense Index
TATA = Total Accruals to Total Assets
LVGI = Leverage Index
```

### Component Calculations

```python
# Days Sales Receivables Index
DSRI = (Receivables_t / Sales_t) / (Receivables_t-1 / Sales_t-1)

# Gross Margin Index
GMI = ((Sales_t-1 - COGS_t-1) / Sales_t-1) / ((Sales_t - COGS_t) / Sales_t)

# Asset Quality Index
AQI = (1 - (CA_t + PPE_t) / TA_t) / (1 - (CA_t-1 + PPE_t-1) / TA_t-1)

# Sales Growth Index
SGI = Sales_t / Sales_t-1

# Depreciation Index
DEPI = (Depreciation_t-1 / (Depreciation_t-1 + PPE_t-1))
     / (Depreciation_t / (Depreciation_t + PPE_t))

# SG&A Expense Index
SGAI = (SGA_t / Sales_t) / (SGA_t-1 / Sales_t-1)

# Total Accruals to Total Assets
TATA = (ΔCA - ΔCash - ΔCL - ΔDebt - Depreciation) / TA

# Leverage Index
LVGI = ((LTD_t + CL_t) / TA_t) / ((LTD_t-1 + CL_t-1) / TA_t-1)
```

### Interpretation

| M-Score | Interpretation |
|---------|----------------|
| < -2.22 | Unlikely manipulator |
| -2.22 to -1.78 | Grey zone |
| > -1.78 | Likely manipulator |

### Red Flags by Component

| Component | High Value Indicates |
|-----------|---------------------|
| DSRI > 1.0 | Revenue recognition issues |
| GMI > 1.0 | Deteriorating margins |
| AQI > 1.0 | Capitalizing expenses |
| SGI > 1.0 | Rapid growth (pressure) |
| DEPI > 1.0 | Slowing depreciation |
| SGAI > 1.0 | Sales pressure |
| TATA > 0.0 | High accruals (quality issue) |
| LVGI > 1.0 | Increasing leverage |

---

## Piotroski F-Score

### Purpose
Assesses financial strength for value investing.

### Components (9 Points Total)

#### Profitability (4 Points)

| Criterion | Condition | Points |
|-----------|-----------|--------|
| ROA | Net Income > 0 | +1 |
| CFO | Operating Cash Flow > 0 | +1 |
| ΔROA | ROA increased YoY | +1 |
| Accruals | CFO > Net Income | +1 |

#### Leverage/Liquidity (3 Points)

| Criterion | Condition | Points |
|-----------|-----------|--------|
| ΔLeverage | Long-term Debt decreased | +1 |
| ΔCurrent | Current Ratio increased | +1 |
| Equity | No new shares issued | +1 |

#### Operating Efficiency (2 Points)

| Criterion | Condition | Points |
|-----------|-----------|--------|
| ΔMargin | Gross Margin increased | +1 |
| ΔTurnover | Asset Turnover increased | +1 |

### Interpretation

| F-Score | Interpretation | Strategy |
|---------|----------------|----------|
| 8-9 | Strong | Buy candidate |
| 7 | Good | Consider buying |
| 5-6 | Neutral | Hold/Monitor |
| 3-4 | Weak | Avoid/Consider selling |
| 0-2 | Very Weak | Strong sell |

### Academic Research

- Original paper: Piotroski (2000)
- High F-Score stocks outperform by ~7.5% annually
- Most effective for small-cap value stocks

---

## Composite Distress Tier

### Purpose
Combines all three scores into a single risk tier.

### Tier Definitions

| Tier | Z-Score | M-Score | F-Score | Valuation Discount |
|------|---------|---------|---------|-------------------|
| **HEALTHY** | ≥ 2.99 | < -2.22 | ≥ 7 | 0% |
| **WATCH** | ≥ 1.81 | < -1.78 | ≥ 5 | 5% |
| **CONCERN** | ≥ 1.23 | < -1.50 | ≥ 3 | 15% |
| **DISTRESSED** | ≥ 0.50 | < -0.80 | ≥ 2 | 30% |
| **SEVERE_DISTRESS** | < 0.50 | ≥ -0.80 | < 2 | 50% |

### Tier Assignment Logic

```python
# Conservative approach: worst score determines tier
# Unless 2+ models agree on a better tier

def assign_tier(z_score, m_score, f_score):
    tier_votes = {tier: 0 for tier in TIERS}

    # Each model votes for a tier
    if z_score >= 2.99:
        tier_votes["HEALTHY"] += 1
    elif z_score >= 1.81:
        tier_votes["WATCH"] += 1
    # ... etc

    # Use worst tier unless majority agrees on better
    worst_tier = get_worst_tier(tier_votes)
    majority_tier = get_majority_tier(tier_votes)

    if tier_votes[majority_tier] >= 2:
        return majority_tier
    return worst_tier
```

### Integration with Valuation

The distress tier applies a discount to fair value estimates:

```
Adjusted Fair Value = Base Fair Value × (1 - Distress Discount)

Example:
Base Fair Value: $100
Distress Tier: CONCERN (15% discount)
Adjusted Fair Value: $100 × (1 - 0.15) = $85
```

---

## CLI Usage

### Individual Models

```bash
# Altman Z-Score
./investigator_v2.sh --val-credit-risk AAPL

# Full credit risk analysis with all models
./investigator_v2.sh --val-integrate AAPL --val-base-fv 190 --val-price 185
```

### Batch Credit Risk Calculation

```bash
# Weekly scheduled job (Sunday 8PM ET)
python scripts/scheduled/calculate_credit_risk.py

# Force recalculation
python scripts/scheduled/calculate_credit_risk.py --force-refresh

# Specific symbols
python scripts/scheduled/calculate_credit_risk.py --symbols AAPL,MSFT,GOOGL
```

### Output Example

```
╔══════════════════════════════════════════════════════════════╗
║  CREDIT RISK ANALYSIS: AAPL                                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ALTMAN Z-SCORE                                              ║
║  ├─ Score:          4.23                                     ║
║  ├─ Zone:           SAFE                                     ║
║  └─ Interpretation: Low bankruptcy risk                      ║
║                                                              ║
║  BENEISH M-SCORE                                             ║
║  ├─ Score:          -2.87                                    ║
║  ├─ Zone:           SAFE                                     ║
║  └─ Interpretation: Unlikely earnings manipulation           ║
║                                                              ║
║  PIOTROSKI F-SCORE                                           ║
║  ├─ Score:          8/9                                      ║
║  ├─ Zone:           STRONG                                   ║
║  └─ Interpretation: Strong financial health                  ║
║                                                              ║
║  COMPOSITE TIER                                              ║
║  ├─ Tier:           HEALTHY                                  ║
║  ├─ Discount:       0%                                       ║
║  └─ Action:         No valuation adjustment needed           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Data Requirements

### SEC XBRL Fields Used

```
# Balance Sheet
Assets
CurrentAssets
Liabilities
CurrentLiabilities
StockholdersEquity
RetainedEarnings
LongTermDebt
PropertyPlantAndEquipmentNet
AccountsReceivable

# Income Statement
Revenues
CostOfRevenue
GrossProfit
OperatingIncome
NetIncome
DepreciationAndAmortization
SellingGeneralAndAdministrative

# Cash Flow
OperatingCashFlow
```

### Calculation Frequency

| Event | Trigger |
|-------|---------|
| New 10-K/10-Q | Automatic recalculation |
| Weekly batch | Sunday 8PM ET |
| On-demand | CLI request |

---

## See Also

- [Free Data Sources](./FREE_DATA_SOURCES.md)
- [Data Collection Schedule](./DATA_COLLECTION_SCHEDULE.md)
- [CLI Data Commands](./CLI_DATA_COMMANDS.md)
