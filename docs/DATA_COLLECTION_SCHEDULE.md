# Data Collection Schedule

This document describes the scheduled data collection jobs that keep Victor-Invest data current.

## Table of Contents

1. [Overview](#overview)
2. [Job Schedule](#job-schedule)
3. [Job Details](#job-details)
4. [Running Jobs](#running-jobs)
5. [Monitoring](#monitoring)

---

## Overview

Victor-Invest uses scheduled jobs to collect and refresh data from various sources. Jobs can be run via:

1. **Cron** (recommended for production)
2. **Python Scheduler** (for development/testing)
3. **Manual CLI** (on-demand)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Scheduler Runner                            │
│                  (scheduler_runner.py)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Treasury   │  │   FRED      │  │   SEC       │             │
│  │  Collector  │  │  Collector  │  │  Collectors │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────────────────────────────────────────────────────┤
│  │                    BaseCollector                            │
│  │  • Logging with rotation                                    │
│  │  • Lock file (prevent concurrent runs)                      │
│  │  • Metrics collection                                       │
│  │  • Retry with exponential backoff                           │
│  │  • Database connection management                           │
│  └─────────────────────────────────────────────────────────────┤
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Job Schedule

### Weekday Schedule (Monday-Friday)

| Time (ET) | Job | Data Source | Description |
|-----------|-----|-------------|-------------|
| 7:00 AM | `collect_13f_filings` | SEC EDGAR | Institutional holdings |
| 8:00 AM | `collect_insider_transactions` | SEC EDGAR | Form 4 transactions |
| 9:00 AM | `refresh_macro_indicators` | FRED API | Macro economic data |
| 12:00 PM | `collect_insider_transactions` | SEC EDGAR | Form 4 transactions |
| 4:00 PM | `collect_insider_transactions` | SEC EDGAR | Form 4 transactions |
| 6:00 PM | `collect_treasury_data` | Treasury API | Yield curve data |
| 6:30 PM | `update_market_regime` | Internal | Market regime classification |
| 8:00 PM | `collect_insider_transactions` | SEC EDGAR | Form 4 transactions |

### Bi-Monthly Schedule

| Day | Time (ET) | Job | Data Source | Description |
|-----|-----------|-----|-------------|-------------|
| 1st | 10:00 AM | `collect_short_interest` | FINRA | Short interest data |
| 15th | 10:00 AM | `collect_short_interest` | FINRA | Short interest data |

### Weekly Schedule

| Day | Time (ET) | Job | Data Source | Description |
|-----|-----------|-----|-------------|-------------|
| Sunday | 8:00 PM | `calculate_credit_risk` | Internal | Credit risk scores |

### Cron Expressions

```cron
# Treasury Yields (Daily 6PM ET, Mon-Fri)
0 18 * * 1-5 /path/to/collect_treasury_data.py

# FRED Macro (Daily 9AM ET, Mon-Fri)
0 9 * * 1-5 /path/to/refresh_macro_indicators.py

# Insider Transactions (Every 4 hours, Mon-Fri)
0 8,12,16,20 * * 1-5 /path/to/collect_insider_transactions.py

# 13F Filings (Daily 7AM ET, Mon-Fri)
0 7 * * 1-5 /path/to/collect_13f_filings.py

# Short Interest (1st and 15th at 10AM ET)
0 10 1,15 * * /path/to/collect_short_interest.py

# Market Regime (Daily 6:30PM ET, Mon-Fri)
30 18 * * 1-5 /path/to/update_market_regime.py

# Credit Risk (Sunday 8PM ET)
0 20 * * 0 /path/to/calculate_credit_risk.py
```

---

## Job Details

### collect_treasury_data.py

**Schedule:** Daily at 6:00 PM ET (after market close)
**Source:** Treasury Fiscal Data API
**Duration:** ~30 seconds

**Collects:**
- Daily yield curve rates (1M to 30Y)
- Calculated spreads (10Y-2Y, 10Y-3M)
- Yield curve inversion detection

**CLI:**
```bash
python scripts/scheduled/collect_treasury_data.py
python scripts/scheduled/collect_treasury_data.py --days 30
```

---

### refresh_macro_indicators.py

**Schedule:** Daily at 9:00 AM ET (before market open)
**Source:** FRED API
**Duration:** ~2 minutes

**Collects:**
- Interest rates (Treasury yields, Fed Funds, Prime)
- Credit spreads (BAA, AAA)
- VIX volatility index
- Economic indicators (GDP, unemployment, CPI)
- Market indices (S&P 500, DJIA, NASDAQ)
- Money supply (M2, Fed balance sheet)
- Housing data (mortgage rates, Case-Shiller)
- Labor data (payrolls, jobless claims)

**Categories:**
- `rates` - Interest rates
- `credit` - Credit spreads
- `volatility` - VIX
- `economic` - GDP, unemployment, inflation
- `markets` - Stock indices
- `money` - Money supply
- `housing` - Housing data
- `labor` - Employment data

**CLI:**
```bash
python scripts/scheduled/refresh_macro_indicators.py
python scripts/scheduled/refresh_macro_indicators.py --category rates
python scripts/scheduled/refresh_macro_indicators.py --days 60
```

---

### collect_insider_transactions.py

**Schedule:** Every 4 hours (8AM, 12PM, 4PM, 8PM ET)
**Source:** SEC EDGAR (Form 4)
**Duration:** ~5 minutes

**Collects:**
- New Form 4 filings
- Insider buy/sell transactions
- Transaction values and share counts
- Insider roles (CEO, CFO, Director, etc.)
- Updates sentiment scores

**CLI:**
```bash
python scripts/scheduled/collect_insider_transactions.py
python scripts/scheduled/collect_insider_transactions.py --symbols AAPL,MSFT
python scripts/scheduled/collect_insider_transactions.py --hours 12
```

---

### collect_13f_filings.py

**Schedule:** Daily at 7:00 AM ET
**Source:** SEC EDGAR (Form 13F)
**Duration:** ~3 minutes

**Collects:**
- New 13F filings from top institutions
- Holdings positions and values
- Quarter-over-quarter changes
- Updates institutional ownership metrics

**Tracked Institutions (20+):**
- Berkshire Hathaway
- BlackRock
- Vanguard
- State Street
- Fidelity (FMR)
- Citadel Advisors
- Bridgewater Associates
- Two Sigma
- Renaissance Technologies
- And more...

**CLI:**
```bash
python scripts/scheduled/collect_13f_filings.py
python scripts/scheduled/collect_13f_filings.py --days 7
python scripts/scheduled/collect_13f_filings.py --top-institutions 50
```

---

### collect_short_interest.py

**Schedule:** 1st and 15th of each month at 10:00 AM ET
**Source:** FINRA
**Duration:** ~3 minutes

**Collects:**
- Short interest positions
- Days to cover calculations
- Short interest ratio changes
- Squeeze potential indicators

**CLI:**
```bash
python scripts/scheduled/collect_short_interest.py
python scripts/scheduled/collect_short_interest.py --symbols GME,AMC
python scripts/scheduled/collect_short_interest.py --threshold 10
```

---

### update_market_regime.py

**Schedule:** Daily at 6:30 PM ET (after treasury data)
**Source:** Internal (uses collected data)
**Duration:** ~30 seconds

**Updates:**
- Current market regime classification
- Credit cycle phase
- Volatility regime
- Recession probability
- Risk-off signals
- Regime transitions

**Dependencies:**
- `collect_treasury_data` (must run first)
- `refresh_macro_indicators` (must run first)

**CLI:**
```bash
python scripts/scheduled/update_market_regime.py
python scripts/scheduled/update_market_regime.py --lookback 60
```

---

### calculate_credit_risk.py

**Schedule:** Weekly on Sunday at 8:00 PM ET
**Source:** Internal (SEC financial data)
**Duration:** ~10 minutes

**Calculates:**
- Altman Z-Score (bankruptcy prediction)
- Beneish M-Score (earnings manipulation)
- Piotroski F-Score (financial strength)
- Composite distress tier

**CLI:**
```bash
python scripts/scheduled/calculate_credit_risk.py
python scripts/scheduled/calculate_credit_risk.py --symbols AAPL,MSFT
python scripts/scheduled/calculate_credit_risk.py --force-refresh
```

---

## Running Jobs

### Using Crontab

```bash
# Generate crontab entries
python scripts/scheduled/generate_crontab.py --show

# Install crontab
python scripts/scheduled/generate_crontab.py --install

# Save to file
python scripts/scheduled/generate_crontab.py --output /tmp/victor_crontab
```

### Using Python Scheduler

```bash
# Run as daemon
python scripts/scheduled/scheduler_runner.py --daemon

# List all jobs
python scripts/scheduled/scheduler_runner.py --list-jobs

# Run specific job now
python scripts/scheduled/scheduler_runner.py --run-now collect_treasury_data

# Run all jobs immediately
python scripts/scheduled/scheduler_runner.py --run-all
```

### Manual Execution

```bash
# Run individual scripts
python scripts/scheduled/collect_treasury_data.py
python scripts/scheduled/refresh_macro_indicators.py
python scripts/scheduled/collect_insider_transactions.py
```

---

## Monitoring

### Job Metrics

Each job records metrics to the `scheduler_job_runs` table:

```sql
SELECT
    job_name,
    start_time,
    duration_seconds,
    records_processed,
    records_inserted,
    success,
    error_count
FROM scheduler_job_runs
WHERE start_time > NOW() - INTERVAL '24 hours'
ORDER BY start_time DESC;
```

### Log Files

Logs are stored in `logs/`:

```
logs/
├── collect_treasury_data.log
├── refresh_macro_indicators.log
├── collect_insider_transactions.log
├── collect_13f_filings.log
├── collect_short_interest.log
├── update_market_regime.log
├── calculate_credit_risk.log
└── scheduler_runner.log
```

### Lock Files

Lock files prevent concurrent execution:

```
logs/locks/
├── collect_treasury_data.lock
├── refresh_macro_indicators.lock
└── ...
```

If a job appears stuck, check for stale lock files:

```bash
# List lock files
ls -la logs/locks/

# Check if process is actually running
cat logs/locks/collect_treasury_data.lock
# Shows PID and start time

# Remove stale lock (only if process is dead)
rm logs/locks/collect_treasury_data.lock
```

### Alerting

Configure alerts in `config/scheduler.yaml`:

```yaml
alerting:
  enabled: true
  email: alerts@example.com
  slack_webhook: https://hooks.slack.com/...

  on_failure: true
  on_warning: true
  on_long_running: true
  long_running_threshold_minutes: 30
```

---

## See Also

- [Free Data Sources](./FREE_DATA_SOURCES.md)
- [Credit Risk Models](./CREDIT_RISK_MODELS.md)
- [CLI Data Commands](./CLI_DATA_COMMANDS.md)
