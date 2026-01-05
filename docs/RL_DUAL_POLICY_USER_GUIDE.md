# RL Dual Policy User Guide

## Overview

The **Dual RL Policy** system combines two specialized reinforcement learning policies for stock valuation and trading decisions:

| Policy | Purpose | Output |
|--------|---------|--------|
| **Technical Policy** | Timing decisions | Long/Short/Skip position signals |
| **Fundamental Policy** | Valuation approach | Model weights, holding periods |

This separation allows each policy to specialize on its domain, improving overall prediction accuracy.

---

## Quick Start

### 1. Run Predictions for Specific Symbols

```bash
cd /Users/vijaysingh/code/victor-invest
source ~/.investigator/env

# Quick prediction for any symbol
PYTHONPATH=./src:. python3 -c "
from investigator.domain.services.rl.policy import DualRLPolicy, TechnicalRLPolicy, FundamentalRLPolicy
from investigator.domain.services.rl.models import ValuationContext, GrowthStage, CompanySize
from datetime import date

# Load trained policies
tech = TechnicalRLPolicy()
fund = FundamentalRLPolicy()
tech.load('data/rl_models/technical_policy.pkl')
fund.load('data/rl_models/fundamental_policy.pkl')
dual = DualRLPolicy(technical_policy=tech, fundamental_policy=fund)

# Create context for your symbol
context = ValuationContext(
    symbol='AAPL',
    analysis_date=date.today(),
    sector='Technology',
    industry='Consumer Electronics',
    growth_stage=GrowthStage.MATURE,
    company_size=CompanySize.MEGA_CAP,
    valuation_gap=-0.10,  # Your estimated gap
    valuation_confidence=0.7,
    # ... other fields with defaults
)

result = dual.predict_full(context)
print(f'Position: {result[\"position\"]}')  # 1=Long, -1=Short, 0=Skip
print(f'Weights: {result[\"weights\"]}')
print(f'Holding: {result[\"holding_period\"]}')
"
```

### 2. Run Full Backtest

```bash
# Run backtest for top 500 stocks
PYTHONPATH=./src:. python3 scripts/rl_backtest.py \
    --top-n 500 \
    --parallel 6 \
    --lookback 12 9 6 3

# Results saved to logs/backtest_results_YYYYMMDD_HHMMSS.json
```

### 3. Train/Retrain Policies

```bash
# Train dual policy from outcomes
PYTHONPATH=./src:. python3 scripts/rl_train_dual.py --epochs 10

# Train contextual bandit policy (alternative)
PYTHONPATH=./src:. python3 scripts/rl_train.py --epochs 50 --deploy
```

---

## Where Predictions Are Stored

### Database: `valuation_outcomes` Table

All predictions are stored in PostgreSQL:

```bash
# Connect to database
PGPASSWORD=investigator psql -h dataserver1.singh.local \
    -U investigator -d sec_database
```

**Key Columns:**

| Column | Description |
|--------|-------------|
| `symbol` | Stock ticker |
| `analysis_date` | Date of prediction |
| `blended_fair_value` | Weighted average fair value |
| `current_price` | Price at prediction time |
| `predicted_upside_pct` | Expected upside % |
| `model_weights` | JSON with DCF, PE, PS, etc. weights |
| `tier_classification` | Company tier (e.g., `high_growth_strong`) |
| `position_type` | `LONG` or `SHORT` |
| `reward_30d`, `reward_90d`, `reward_365d` | Actual returns |

**Query Examples:**

```sql
-- Latest predictions
SELECT symbol, analysis_date, predicted_upside_pct, tier_classification
FROM valuation_outcomes
ORDER BY analysis_date DESC, predicted_upside_pct DESC
LIMIT 20;

-- Best performing tiers
SELECT tier_classification,
       COUNT(*) as predictions,
       AVG(reward_90d) as avg_reward
FROM valuation_outcomes
WHERE reward_90d IS NOT NULL
GROUP BY tier_classification
ORDER BY avg_reward DESC;

-- Get all LONG recommendations with >20% upside
SELECT symbol, analysis_date, predicted_upside_pct, blended_fair_value, current_price
FROM valuation_outcomes
WHERE position_type = 'LONG'
  AND predicted_upside_pct > 20
  AND analysis_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY predicted_upside_pct DESC;
```

### File Storage

| Path | Contents |
|------|----------|
| `data/rl_models/technical_policy.pkl` | Technical policy (timing) |
| `data/rl_models/fundamental_policy.pkl` | Fundamental policy (weights) |
| `data/rl_models/policy.pkl` | Contextual bandit policy |
| `data/rl_models/normalizer.pkl` | Feature normalizer |
| `data/rl_models/training_log.json` | Training metrics |
| `logs/backtest_results_*.json` | Backtest results |

---

## Weekly Automation

### Option 1: Cron Job (Recommended)

Create `/Users/vijaysingh/code/victor-invest/scripts/weekly_rl_pipeline.sh`:

```bash
#!/bin/bash
# Weekly RL Pipeline - Run every Sunday at 6 AM

set -e

cd /Users/vijaysingh/code/victor-invest
source ~/.investigator/env

LOG_DIR="logs/weekly"
mkdir -p $LOG_DIR
DATE=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/weekly_pipeline_$DATE.log"

echo "=== Weekly RL Pipeline Started: $(date) ===" | tee -a $LOG_FILE

# Step 1: Update outcome prices (get actual returns for past predictions)
echo "[1/4] Updating outcome prices..." | tee -a $LOG_FILE
PYTHONPATH=./src:. python3 scripts/rl_outcome_updater.py >> $LOG_FILE 2>&1

# Step 2: Retrain policies with new outcomes
echo "[2/4] Retraining RL policies..." | tee -a $LOG_FILE
PYTHONPATH=./src:. python3 scripts/rl_train.py --epochs 50 --deploy >> $LOG_FILE 2>&1

# Step 3: Run backtest on top 500 stocks
echo "[3/4] Running weekly backtest..." | tee -a $LOG_FILE
PYTHONPATH=./src:. python3 scripts/rl_backtest.py \
    --top-n 500 \
    --parallel 6 \
    --lookback 3 6 12 >> $LOG_FILE 2>&1

# Step 4: Generate weekly recommendations report
echo "[4/4] Generating recommendations..." | tee -a $LOG_FILE
PYTHONPATH=./src:. python3 scripts/generate_weekly_recommendations.py >> $LOG_FILE 2>&1

echo "=== Weekly RL Pipeline Completed: $(date) ===" | tee -a $LOG_FILE

# Optional: Send notification
# curl -X POST "https://your-webhook-url" -d "Weekly RL pipeline completed"
```

**Add to crontab:**

```bash
# Edit crontab
crontab -e

# Add this line (runs every Sunday at 6:00 AM)
0 6 * * 0 /Users/vijaysingh/code/victor-invest/scripts/weekly_rl_pipeline.sh
```

### Option 2: LaunchAgent (macOS)

Create `~/Library/LaunchAgents/com.victor.rl-weekly.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.victor.rl-weekly</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/vijaysingh/code/victor-invest/scripts/weekly_rl_pipeline.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>6</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/vijaysingh/code/victor-invest/logs/weekly/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/vijaysingh/code/victor-invest/logs/weekly/launchd.err</string>
</dict>
</plist>
```

**Load the agent:**

```bash
launchctl load ~/Library/LaunchAgents/com.victor.rl-weekly.plist
```

---

## Policy Interpretation

### Technical Policy Learnings

From 125,250 training samples:

| Action | Samples | Avg Reward | Interpretation |
|--------|---------|------------|----------------|
| SKIP | 6,090 | +0.0199 | Best reward - learned to be selective |
| SHORT | 81,660 | -0.0075 | Underperforms - higher bar applied |
| LONG | 37,500 | -0.0170 | Modest underperformance |

**Key Asymmetries:**
- LONG: Requires >5% valuation gap
- SHORT: Requires >15% valuation gap (higher bar)
- Confidence threshold: 60%
- High confidence (>80%) is penalized (overconfidence often wrong)

### Fundamental Policy Learnings

**Model Performance Ranking:**

| Model | Avg Reward | Interpretation |
|-------|------------|----------------|
| GGM (Gordon Growth) | +0.0066 | Best - dividend models most reliable |
| PB (Price/Book) | 0.0000 | Neutral |
| PE (Price/Earnings) | -0.0018 | Slight underperformance |
| EV/EBITDA | -0.0023 | Slight underperformance |
| PS (Price/Sales) | -0.0030 | Underperforms |
| DCF | -0.0035 | Most complex, least reliable |

**Holding Period:**
- 1 month: +0.0010 (positive)
- 3 months: -0.0090 (negative)

**Interpretation:** Shorter holding periods are more predictive.

### Sector-Specific Insights

From backtest analysis:

| Sector/Tier | Avg Reward | Recommendation |
|-------------|------------|----------------|
| Semiconductor | -0.141 | Avoid (high volatility) |
| REIT Core | +0.059 | Favorable |
| Insurance High Quality | +0.065 | Favorable |
| Dividend Aristocrat | +0.051 | Favorable |
| Pre-Profit | -0.039 | Cautious |

---

## Generating Weekly Recommendations

Create `scripts/generate_weekly_recommendations.py`:

```python
#!/usr/bin/env python3
"""Generate weekly stock recommendations from RL predictions."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from investigator.infrastructure.database.db import DatabaseEngine

def generate_recommendations():
    engine = DatabaseEngine().engine

    # Get predictions from last 7 days
    query = """
    SELECT
        symbol,
        analysis_date,
        blended_fair_value,
        current_price,
        predicted_upside_pct,
        tier_classification,
        position_type,
        model_weights
    FROM valuation_outcomes
    WHERE analysis_date >= CURRENT_DATE - INTERVAL '7 days'
      AND predicted_upside_pct IS NOT NULL
    ORDER BY predicted_upside_pct DESC
    """

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()

    # Categorize recommendations
    longs = [r for r in rows if r['position_type'] == 'LONG' and r['predicted_upside_pct'] > 15]
    shorts = [r for r in rows if r['position_type'] == 'SHORT' and r['predicted_upside_pct'] < -20]

    # Generate report
    report = {
        "generated_at": datetime.now().isoformat(),
        "period": "Last 7 days",
        "summary": {
            "total_predictions": len(rows),
            "long_recommendations": len(longs),
            "short_recommendations": len(shorts),
        },
        "top_longs": [
            {
                "symbol": r['symbol'],
                "upside": float(r['predicted_upside_pct']),
                "fair_value": float(r['blended_fair_value']),
                "current_price": float(r['current_price']),
                "tier": r['tier_classification'],
            }
            for r in longs[:10]
        ],
        "top_shorts": [
            {
                "symbol": r['symbol'],
                "downside": float(r['predicted_upside_pct']),
                "fair_value": float(r['blended_fair_value']),
                "current_price": float(r['current_price']),
                "tier": r['tier_classification'],
            }
            for r in shorts[:10]
        ],
    }

    # Save report
    output_path = Path("logs/weekly") / f"recommendations_{datetime.now():%Y%m%d}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Recommendations saved to: {output_path}")
    print(f"Top LONGs: {[r['symbol'] for r in report['top_longs']]}")
    print(f"Top SHORTs: {[r['symbol'] for r in report['top_shorts']]}")

if __name__ == "__main__":
    generate_recommendations()
```

---

## CLI Commands Reference

```bash
# === BACKTEST ===
# Full backtest (500 stocks, 4 lookback periods)
PYTHONPATH=./src:. python3 scripts/rl_backtest.py --top-n 500 --parallel 6 --lookback 12 9 6 3

# Quick backtest (specific symbols)
PYTHONPATH=./src:. python3 scripts/rl_backtest.py --symbols AAPL NVDA MSFT --lookback 3 6

# === TRAINING ===
# Train contextual bandit policy
PYTHONPATH=./src:. python3 scripts/rl_train.py --epochs 50 --deploy

# Train dual policy (technical + fundamental)
PYTHONPATH=./src:. python3 scripts/rl_train_dual.py --epochs 10

# === OUTCOMES ===
# Update outcome prices (get actual returns)
PYTHONPATH=./src:. python3 scripts/rl_outcome_updater.py

# === DATABASE ===
# View recent predictions
PGPASSWORD=investigator psql -h dataserver1.singh.local -U investigator -d sec_database \
    -c "SELECT symbol, predicted_upside_pct, tier_classification FROM valuation_outcomes ORDER BY analysis_date DESC LIMIT 20;"

# Count predictions by tier
PGPASSWORD=investigator psql -h dataserver1.singh.local -U investigator -d sec_database \
    -c "SELECT tier_classification, COUNT(*) FROM valuation_outcomes GROUP BY tier_classification ORDER BY count DESC;"
```

---

## Troubleshooting

### Common Issues

**1. Policy not loading:**
```bash
# Check if policy files exist
ls -la data/rl_models/*.pkl

# Retrain if missing
PYTHONPATH=./src:. python3 scripts/rl_train.py --epochs 50 --deploy
```

**2. Database connection failed:**
```bash
# Check environment variables
cat ~/.investigator/env

# Test connection
PGPASSWORD=investigator psql -h dataserver1.singh.local -U investigator -d sec_database -c "SELECT 1;"
```

**3. No predictions generated:**
```bash
# Check backtest logs
tail -100 rl_backtest_*.log | grep -E "(Error|Processing|Completed)"

# Check for missing data
PGPASSWORD=investigator psql -h dataserver1.singh.local -U investigator -d sec_database \
    -c "SELECT COUNT(*) FROM valuation_outcomes WHERE analysis_date >= CURRENT_DATE - 7;"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DUAL RL POLICY SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐          ┌──────────────────┐             │
│  │ Technical Policy │          │ Fundamental Policy│             │
│  │   (Timing)       │          │   (Valuation)    │             │
│  ├──────────────────┤          ├──────────────────┤             │
│  │ • RSI, MACD      │          │ • Sector         │             │
│  │ • ADX, Stoch     │          │ • Industry       │             │
│  │ • MFI, OBV       │          │ • Growth Stage   │             │
│  │ • Valuation Gap  │          │ • Profitability  │             │
│  └────────┬─────────┘          └────────┬─────────┘             │
│           │                             │                        │
│           ▼                             ▼                        │
│  ┌──────────────────┐          ┌──────────────────┐             │
│  │ Position Signal  │          │ Model Weights    │             │
│  │ Long/Short/Skip  │          │ DCF, PE, PS, PB  │             │
│  │ + Confidence     │          │ EV/EBITDA, GGM   │             │
│  └────────┬─────────┘          └────────┬─────────┘             │
│           │                             │                        │
│           └──────────┬──────────────────┘                        │
│                      ▼                                           │
│             ┌──────────────────┐                                 │
│             │ Combined         │                                 │
│             │ Recommendation   │                                 │
│             │ + Holding Period │                                 │
│             └────────┬─────────┘                                 │
│                      ▼                                           │
│             ┌──────────────────┐                                 │
│             │ valuation_       │                                 │
│             │ outcomes (DB)    │                                 │
│             └──────────────────┘                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-03 | Initial dual policy system |
| 1.1 | 2026-01-03 | Asymmetric thresholds for Long/Short |

---

*Last Updated: 2026-01-03*
