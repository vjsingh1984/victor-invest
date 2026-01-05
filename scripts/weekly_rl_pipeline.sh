#!/bin/bash
# Weekly RL Pipeline - Run every Sunday at 6 AM
#
# Usage:
#   ./scripts/weekly_rl_pipeline.sh           # Run full pipeline
#   ./scripts/weekly_rl_pipeline.sh --quick   # Skip backtest (faster)
#
# Cron setup:
#   crontab -e
#   0 6 * * 0 /Users/vijaysingh/code/victor-invest/scripts/weekly_rl_pipeline.sh

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/weekly"
DATE=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/weekly_pipeline_$DATE.log"

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
fi

# Setup
cd "$PROJECT_DIR"
source ~/.investigator/env 2>/dev/null || true
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================"
log "Weekly RL Pipeline Started"
log "========================================"
log "Project: $PROJECT_DIR"
log "Quick Mode: $QUICK_MODE"

# Step 1: Update outcome prices
log ""
log "[1/4] Updating outcome prices..."
if PYTHONPATH=./src:. python3 scripts/rl_outcome_updater.py >> "$LOG_FILE" 2>&1; then
    log "  ✓ Outcome prices updated"
else
    log "  ⚠ Outcome update had issues (continuing...)"
fi

# Step 2: Retrain policies
log ""
log "[2/4] Retraining RL policies..."
if PYTHONPATH=./src:. python3 scripts/rl_train.py --epochs 50 --deploy >> "$LOG_FILE" 2>&1; then
    log "  ✓ Policies retrained and deployed"
else
    log "  ✗ Policy training failed"
    exit 1
fi

# Step 3: Run backtest (skip in quick mode)
if [[ "$QUICK_MODE" == false ]]; then
    log ""
    log "[3/4] Running weekly backtest (top 500 stocks)..."
    if PYTHONPATH=./src:. python3 scripts/rl_backtest.py \
        --top-n 500 \
        --parallel 6 \
        --lookback 3 6 12 >> "$LOG_FILE" 2>&1; then
        log "  ✓ Backtest completed"
    else
        log "  ⚠ Backtest had issues (continuing...)"
    fi
else
    log ""
    log "[3/4] Skipping backtest (quick mode)"
fi

# Step 4: Generate recommendations
log ""
log "[4/4] Generating recommendations..."
if PYTHONPATH=./src:. python3 scripts/generate_weekly_recommendations.py >> "$LOG_FILE" 2>&1; then
    log "  ✓ Recommendations generated"
else
    log "  ⚠ Recommendations generation had issues"
fi

# Summary
log ""
log "========================================"
log "Weekly RL Pipeline Completed"
log "========================================"

# Show recent predictions count
PREDICTIONS=$(PGPASSWORD=investigator psql -h dataserver1.singh.local \
    -U investigator -d sec_database -t -c \
    "SELECT COUNT(*) FROM valuation_outcomes WHERE analysis_date >= CURRENT_DATE - 7;" 2>/dev/null || echo "N/A")
log "Predictions (last 7 days): $PREDICTIONS"

# Show latest recommendation file
LATEST_REC=$(ls -t "$LOG_DIR"/recommendations_*.json 2>/dev/null | head -1)
if [[ -n "$LATEST_REC" ]]; then
    log "Latest recommendations: $LATEST_REC"
fi

log ""
log "Log file: $LOG_FILE"
