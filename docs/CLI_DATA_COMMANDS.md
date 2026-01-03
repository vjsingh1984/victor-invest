# CLI Data Commands Reference

This document provides a comprehensive reference for all data-related CLI commands in Victor-Invest.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Macro Data (FRED)](#macro-data-fred)
3. [Insider Trading (SEC Form 4)](#insider-trading-sec-form-4)
4. [Treasury Data](#treasury-data)
5. [Institutional Holdings (SEC 13F)](#institutional-holdings-sec-13f)
6. [Short Interest (FINRA)](#short-interest-finra)
7. [Market Regime](#market-regime)
8. [Valuation Signals](#valuation-signals)
9. [Output Formats](#output-formats)

---

## Quick Reference

```bash
# Macro Data
./investigator_v2.sh --macro-summary
./investigator_v2.sh --macro-buffett
./investigator_v2.sh --macro-category rates

# Insider Trading
./investigator_v2.sh --insider-sentiment AAPL
./investigator_v2.sh --insider-recent AAPL
./investigator_v2.sh --insider-clusters AAPL

# Treasury Data
./investigator_v2.sh --treasury-curve
./investigator_v2.sh --treasury-spread
./investigator_v2.sh --treasury-recession

# Institutional Holdings
./investigator_v2.sh --inst-holdings AAPL
./investigator_v2.sh --inst-top-holders AAPL
./investigator_v2.sh --inst-changes AAPL

# Short Interest
./investigator_v2.sh --short-current GME
./investigator_v2.sh --short-squeeze GME
./investigator_v2.sh --short-most-shorted

# Market Regime
./investigator_v2.sh --regime-summary
./investigator_v2.sh --regime-credit-cycle
./investigator_v2.sh --regime-recommendations

# Valuation Signals
./investigator_v2.sh --val-integrate AAPL --val-base-fv 190 --val-price 185
./investigator_v2.sh --val-credit-risk AAPL
```

---

## Macro Data (FRED)

Access Federal Reserve Economic Data (FRED) indicators.

### Commands

| Command | Description |
|---------|-------------|
| `--macro-summary` | Comprehensive summary of all macro indicators |
| `--macro-buffett` | Buffett Indicator (Market Cap / GDP ratio) |
| `--macro-category CATEGORY` | Get indicators for a specific category |
| `--macro-indicators ID,ID` | Get specific indicators by FRED series ID |
| `--macro-time-series ID` | Get historical time series for an indicator |
| `--macro-list-categories` | List all available categories |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--macro-lookback-days` | 1095 | Days of historical data (default: 3 years) |
| `--macro-limit` | 1000 | Maximum data points for time series |
| `--macro-json` | false | Output raw JSON instead of formatted table |

### Categories

```bash
# Available categories
rates       # Interest rates & yields
inflation   # CPI, PCE, breakeven rates
growth      # GDP, industrial production
employment  # Unemployment, payrolls, claims
credit      # Credit spreads, yields
volatility  # VIX, market volatility
housing     # Mortgage rates, home prices
consumer    # Consumer sentiment, spending
```

### Examples

```bash
# Get all macro indicators with Buffett Indicator
./investigator_v2.sh --macro-summary

# Get just the Buffett Indicator
./investigator_v2.sh --macro-buffett

# Get interest rate indicators
./investigator_v2.sh --macro-category rates

# Get specific indicators
./investigator_v2.sh --macro-indicators DGS10,FEDFUNDS,VIXCLS

# Get 10Y Treasury history (1 year)
./investigator_v2.sh --macro-time-series DGS10 --macro-lookback-days 365

# List all categories and their indicators
./investigator_v2.sh --macro-list-categories

# Output as JSON for scripting
./investigator_v2.sh --macro-summary --macro-json
```

---

## Insider Trading (SEC Form 4)

Access SEC Form 4 insider transaction data.

### Commands

| Command | Description |
|---------|-------------|
| `--insider-sentiment SYMBOL` | Get insider sentiment analysis (buy/sell ratio) |
| `--insider-recent SYMBOL` | Get recent insider transactions |
| `--insider-clusters SYMBOL` | Detect coordinated insider buying/selling |
| `--insider-key-insiders SYMBOL` | Get C-suite and director activity summary |
| `--insider-fetch SYMBOL` | Fetch latest Form 4 filings from SEC EDGAR |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--insider-days` | 90 | Analysis period in days |
| `--insider-significant-only` | false | Show only significant transactions |
| `--insider-json` | false | Output raw JSON |

### Examples

```bash
# Get insider sentiment for Apple
./investigator_v2.sh --insider-sentiment AAPL

# Get recent transactions (last 60 days)
./investigator_v2.sh --insider-recent AAPL --insider-days 60

# Detect cluster buying/selling
./investigator_v2.sh --insider-clusters NVDA

# Get C-suite activity
./investigator_v2.sh --insider-key-insiders TSLA

# Fetch fresh data from SEC
./investigator_v2.sh --insider-fetch MSFT

# Only significant transactions
./investigator_v2.sh --insider-recent AAPL --insider-significant-only

# JSON output for scripting
./investigator_v2.sh --insider-sentiment AAPL --insider-json
```

### Transaction Types

```
P  - Open market purchase
S  - Open market sale
A  - Grant/Award
D  - Disposition (sale) to issuer
M  - Exercise of derivative
F  - Tax withholding
G  - Gift
```

---

## Treasury Data

Access U.S. Treasury yield curve data and market regime analysis.

### Commands

| Command | Description |
|---------|-------------|
| `--treasury-curve` | Get current Treasury yield curve (1M to 30Y) |
| `--treasury-spread` | Get yield spread analysis (10Y-2Y, 10Y-3M) |
| `--treasury-regime` | Get market regime from yield curve shape |
| `--treasury-recession` | Get recession probability & economic phase |
| `--treasury-summary` | Get comprehensive market regime summary |
| `--treasury-history` | Get historical yield data |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--treasury-days` | 365 | Days for historical data |
| `--treasury-maturity` | 10y | Maturity for history (1m, 3m, 1y, 2y, 5y, 10y, 30y) |
| `--treasury-json` | false | Output raw JSON |

### Examples

```bash
# Get current yield curve
./investigator_v2.sh --treasury-curve

# Get yield spread analysis
./investigator_v2.sh --treasury-spread

# Get yield curve regime (shape)
./investigator_v2.sh --treasury-regime

# Get recession probability
./investigator_v2.sh --treasury-recession

# Get comprehensive summary
./investigator_v2.sh --treasury-summary

# Get 10Y yield history (1 year)
./investigator_v2.sh --treasury-history --treasury-maturity 10y --treasury-days 365

# Get 2Y yield history (2 years)
./investigator_v2.sh --treasury-history --treasury-maturity 2y --treasury-days 730

# JSON output
./investigator_v2.sh --treasury-curve --treasury-json
```

### Yield Curve Shapes

```
STEEP           - Normal, healthy economy
NORMAL          - Standard upward slope
FLAT            - Uncertainty, transition
INVERTED        - Recession warning (10Y < 2Y)
DEEPLY_INVERTED - Strong recession signal
```

---

## Institutional Holdings (SEC 13F)

Access SEC Form 13F institutional holdings data.

### Commands

| Command | Description |
|---------|-------------|
| `--inst-holdings SYMBOL` | Get institutional ownership summary |
| `--inst-top-holders SYMBOL` | Get top institutional holders by value |
| `--inst-changes SYMBOL` | Get ownership changes over quarters |
| `--inst-institution CIK` | Get holdings for specific institution by CIK |
| `--inst-search QUERY` | Search for institutions by name |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--inst-limit` | 20 | Number of results to return |
| `--inst-quarters` | 4 | Quarters for change analysis |
| `--inst-quarter` | - | Specific quarter (e.g., '2024-Q4') |
| `--inst-json` | false | Output raw JSON |

### Examples

```bash
# Get institutional ownership summary
./investigator_v2.sh --inst-holdings AAPL

# Get top 25 holders
./investigator_v2.sh --inst-top-holders AAPL --inst-limit 25

# Get ownership changes over 8 quarters
./investigator_v2.sh --inst-changes NVDA --inst-quarters 8

# Get Berkshire Hathaway's holdings
./investigator_v2.sh --inst-institution 0001067983

# Search for institutions containing "vanguard"
./investigator_v2.sh --inst-search "vanguard"

# Get specific quarter
./investigator_v2.sh --inst-holdings AAPL --inst-quarter "2024-Q4"

# JSON output
./investigator_v2.sh --inst-holdings AAPL --inst-json
```

### Notable Institution CIKs

```
0001067983 - Berkshire Hathaway
0001350694 - BlackRock
0001166559 - Vanguard Group
0001364742 - State Street
0001037389 - Fidelity (FMR)
0001536411 - Citadel Advisors
0001061768 - Renaissance Technologies
0001167483 - JPMorgan Chase
```

---

## Short Interest (FINRA)

Access FINRA short interest data.

### Commands

| Command | Description |
|---------|-------------|
| `--short-current SYMBOL` | Get current short interest |
| `--short-history SYMBOL` | Get historical short interest |
| `--short-volume SYMBOL` | Get daily short volume |
| `--short-squeeze SYMBOL` | Calculate short squeeze risk |
| `--short-most-shorted` | Get list of most shorted stocks |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--short-periods` | 12 | Periods for historical data |
| `--short-days` | 30 | Days for volume data |
| `--short-limit` | 20 | Number of stocks for most-shorted |
| `--short-json` | false | Output raw JSON |

### Examples

```bash
# Get current short interest
./investigator_v2.sh --short-current GME

# Get 12 periods of history
./investigator_v2.sh --short-history AMC --short-periods 12

# Get 30 days of short volume
./investigator_v2.sh --short-volume AAPL --short-days 30

# Assess squeeze risk
./investigator_v2.sh --short-squeeze GME

# Get top 25 most shorted stocks
./investigator_v2.sh --short-most-shorted --short-limit 25

# JSON output
./investigator_v2.sh --short-current TSLA --short-json
```

### Squeeze Risk Factors

```
Days to Cover > 5     - Potential squeeze setup
Short % Float > 20%   - High short interest
Rising Short Interest - Building bearish pressure
High Borrow Cost      - Shorts under pressure
```

---

## Market Regime

Access market regime detection and analysis.

### Commands

| Command | Description |
|---------|-------------|
| `--regime-summary` | Get comprehensive market regime summary |
| `--regime-credit-cycle` | Get credit cycle phase analysis |
| `--regime-yield-curve` | Get yield curve shape and signals |
| `--regime-recession` | Get recession probability assessment |
| `--regime-volatility` | Get volatility regime classification |
| `--regime-recommendations` | Get sector allocation recommendations |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--regime-json` | false | Output raw JSON |

### Examples

```bash
# Get comprehensive market regime
./investigator_v2.sh --regime-summary

# Get credit cycle phase
./investigator_v2.sh --regime-credit-cycle

# Get yield curve analysis
./investigator_v2.sh --regime-yield-curve

# Get recession probability
./investigator_v2.sh --regime-recession

# Get volatility regime
./investigator_v2.sh --regime-volatility

# Get investment recommendations
./investigator_v2.sh --regime-recommendations

# JSON output
./investigator_v2.sh --regime-summary --regime-json
```

### Regime Classifications

```
Credit Cycle Phases:
  EARLY_EXPANSION  - Recovery, risk-on
  MID_CYCLE        - Normal growth
  LATE_CYCLE       - Peak, caution
  CREDIT_CRISIS    - Risk-off, defensive

Volatility Regimes:
  LOW              - VIX < 15
  NORMAL           - VIX 15-20
  ELEVATED         - VIX 20-30
  HIGH             - VIX 30-40
  EXTREME          - VIX > 40
```

---

## Valuation Signals

Integrate multiple data signals into valuation adjustments.

### Commands

| Command | Description |
|---------|-------------|
| `--val-integrate SYMBOL` | Full signal integration for adjusted fair value |
| `--val-credit-risk SYMBOL` | Credit risk signal only (Z, M, F scores) |
| `--val-insider SYMBOL` | Insider sentiment signal only |
| `--val-short-interest SYMBOL` | Short interest signal only |
| `--val-market-regime` | Market regime adjustment (no symbol needed) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--val-base-fv VALUE` | Required | Base fair value (for --val-integrate) |
| `--val-price VALUE` | Required | Current price (for --val-integrate) |
| `--val-json` | false | Output raw JSON |

### Examples

```bash
# Full integration with base fair value and current price
./investigator_v2.sh --val-integrate AAPL --val-base-fv 190 --val-price 185

# Credit risk signal only
./investigator_v2.sh --val-credit-risk AAPL

# Insider sentiment signal
./investigator_v2.sh --val-insider AAPL

# Short interest signal
./investigator_v2.sh --val-short-interest GME

# Market regime adjustment
./investigator_v2.sh --val-market-regime

# JSON output
./investigator_v2.sh --val-integrate AAPL --val-base-fv 190 --val-price 185 --val-json
```

### Distress Tiers

| Tier | Discount | Trigger |
|------|----------|---------|
| HEALTHY | 0% | Z > 2.99, M < -2.22, F ≥ 7 |
| WATCH | 5% | Z > 1.81, M < -1.78, F ≥ 5 |
| CONCERN | 15% | Z > 1.23, M < -1.50, F ≥ 3 |
| DISTRESSED | 30% | Z > 0.50, M < -0.80, F ≥ 2 |
| SEVERE_DISTRESS | 50% | Z < 0.50 or M > -0.80 or F < 2 |

---

## Output Formats

### Table Format (Default)

Human-readable formatted tables with visual indicators.

```bash
./investigator_v2.sh --treasury-curve
```

```
╔══════════════════════════════════════════════════════════════╗
║  TREASURY YIELD CURVE                                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Maturity    Yield     Change    Visual                     ║
║  ─────────────────────────────────────────────              ║
║  1-Month     4.35%     +0.02     ████████████               ║
║  3-Month     4.42%     +0.01     █████████████              ║
║  2-Year      4.28%     -0.03     ███████████                ║
║  10-Year     4.15%     -0.05     ██████████                 ║
║  30-Year     4.32%     -0.02     ████████████               ║
║                                                              ║
║  10Y-2Y Spread: -0.13% (INVERTED)                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### JSON Format

Machine-readable JSON for scripting and integration.

```bash
./investigator_v2.sh --treasury-curve --treasury-json
```

```json
{
  "status": "success",
  "data": {
    "yields": {
      "1m": 4.35,
      "3m": 4.42,
      "2y": 4.28,
      "10y": 4.15,
      "30y": 4.32
    },
    "spreads": {
      "10y_2y": -0.13,
      "10y_3m": -0.27
    },
    "inverted": true,
    "timestamp": "2025-01-02T18:00:00Z"
  }
}
```

### Using JSON Output in Scripts

```bash
# Get specific value with jq
./investigator_v2.sh --treasury-curve --treasury-json | jq '.data.spreads["10y_2y"]'

# Check for inversion
./investigator_v2.sh --treasury-curve --treasury-json | jq '.data.inverted'

# Store in variable
SPREAD=$(./investigator_v2.sh --treasury-spread --treasury-json | jq -r '.data.spreads["10y_2y"]')
```

---

## See Also

- [Free Data Sources](./FREE_DATA_SOURCES.md)
- [Credit Risk Models](./CREDIT_RISK_MODELS.md)
- [Data Collection Schedule](./DATA_COLLECTION_SCHEDULE.md)
