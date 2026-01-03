-- InvestiGator RL (Reinforcement Learning) Tables
-- Version: 7.0.0
-- RDBMS-Agnostic: Works with PostgreSQL and SQLite
-- Copyright (c) 2025 Vijaykumar Singh
-- Licensed under Apache License 2.0
--
-- Tables for: valuation_outcomes, rl_training, credit_risk

-- ============================================================================
-- VALUATION OUTCOMES (RL Training Data)
-- ============================================================================

CREATE TABLE IF NOT EXISTS valuation_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    analysis_date TEXT NOT NULL,
    fiscal_period TEXT,

    -- Prediction Data
    blended_fair_value REAL,
    current_price REAL,
    predicted_upside_pct REAL,

    -- Individual Model Fair Values
    dcf_fair_value REAL,
    pe_fair_value REAL,
    ps_fair_value REAL,
    evebitda_fair_value REAL,
    pb_fair_value REAL,
    ggm_fair_value REAL,

    -- Weights and Context
    model_weights TEXT,  -- JSON
    tier_classification TEXT,
    context_features TEXT,  -- JSON with sector, industry, growth_stage, etc.

    -- Entry/Exit Dates
    entry_date TEXT,
    exit_date_30d TEXT,
    exit_date_90d TEXT,
    exit_date_365d TEXT,

    -- Outcome Data (updated after time passes)
    actual_price_30d REAL,
    actual_price_90d REAL,
    actual_price_365d REAL,
    outcome_updated_at TEXT,

    -- Calculated Rewards
    reward_30d REAL,
    reward_90d REAL,
    reward_365d REAL,

    -- Per-Model Rewards
    per_model_rewards TEXT,  -- JSON

    -- RL Training Metadata
    used_for_training INTEGER DEFAULT 0,
    training_batch_id INTEGER,

    -- A/B Test Tracking
    ab_test_group TEXT,
    policy_version TEXT,

    -- Position type for dual policy
    position_type TEXT DEFAULT 'inferred',

    -- Metadata
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),

    UNIQUE(symbol, analysis_date, position_type)
);

CREATE INDEX IF NOT EXISTS idx_outcomes_symbol ON valuation_outcomes(symbol);
CREATE INDEX IF NOT EXISTS idx_outcomes_date ON valuation_outcomes(analysis_date);
CREATE INDEX IF NOT EXISTS idx_outcomes_tier ON valuation_outcomes(tier_classification);
CREATE INDEX IF NOT EXISTS idx_outcomes_ab_test ON valuation_outcomes(ab_test_group);
CREATE INDEX IF NOT EXISTS idx_outcomes_training ON valuation_outcomes(used_for_training);
CREATE INDEX IF NOT EXISTS idx_outcomes_position ON valuation_outcomes(position_type);

-- ============================================================================
-- RL TRAINING BATCHES
-- ============================================================================

CREATE TABLE IF NOT EXISTS rl_training_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_date TEXT NOT NULL,

    -- Training Configuration
    policy_type TEXT NOT NULL,
    config_snapshot TEXT,  -- JSON

    -- Training Data Stats
    num_experiences INTEGER,
    train_size INTEGER,
    validation_size INTEGER,
    test_size INTEGER,

    -- Training Metrics
    train_loss REAL,
    validation_loss REAL,
    train_reward_mean REAL,
    validation_reward_mean REAL,

    -- Model Artifact
    model_path TEXT,
    model_version TEXT,

    -- Evaluation Metrics
    baseline_mape REAL,
    rl_mape REAL,
    mape_improvement_pct REAL,
    direction_accuracy REAL,

    -- Status
    status TEXT DEFAULT 'completed',
    error_message TEXT,

    -- Metadata
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_batches_date ON rl_training_batches(batch_date);
CREATE INDEX IF NOT EXISTS idx_training_batches_policy ON rl_training_batches(policy_type);

-- ============================================================================
-- RL POLICY METRICS
-- ============================================================================

CREATE TABLE IF NOT EXISTS rl_policy_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_date TEXT NOT NULL,

    -- Grouping Dimensions
    sector TEXT,
    industry TEXT,
    tier_classification TEXT,

    -- Aggregated Performance
    num_predictions INTEGER,
    avg_reward_30d REAL,
    avg_reward_90d REAL,
    avg_error_pct REAL,
    direction_accuracy REAL,

    -- Model-Specific Metrics
    model_performance TEXT,  -- JSON

    -- A/B Test Results
    rl_avg_reward REAL,
    baseline_avg_reward REAL,
    statistical_significance REAL,

    -- Metadata
    created_at TEXT DEFAULT (datetime('now')),

    UNIQUE(metric_date, sector, industry, tier_classification)
);

CREATE INDEX IF NOT EXISTS idx_policy_metrics_date ON rl_policy_metrics(metric_date);
CREATE INDEX IF NOT EXISTS idx_policy_metrics_sector ON rl_policy_metrics(sector);

-- ============================================================================
-- CREDIT RISK SCORES
-- ============================================================================

CREATE TABLE IF NOT EXISTS credit_risk_scores (
    symbol TEXT NOT NULL,
    calculation_date TEXT NOT NULL,
    altman_z_score REAL,
    altman_z_interpretation TEXT,
    beneish_m_score REAL,
    beneish_m_interpretation TEXT,
    piotroski_f_score INTEGER,
    piotroski_f_interpretation TEXT,
    distress_tier TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, calculation_date)
);

CREATE INDEX IF NOT EXISTS idx_credit_risk_symbol ON credit_risk_scores(symbol);
CREATE INDEX IF NOT EXISTS idx_credit_risk_date ON credit_risk_scores(calculation_date DESC);

CREATE TABLE IF NOT EXISTS current_credit_risk (
    symbol TEXT PRIMARY KEY,
    altman_z_score REAL,
    beneish_m_score REAL,
    piotroski_f_score INTEGER,
    distress_tier TEXT,
    last_updated TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS credit_risk_summary (
    summary_date TEXT PRIMARY KEY,
    total_analyzed INTEGER,
    healthy_count INTEGER DEFAULT 0,
    watch_count INTEGER DEFAULT 0,
    concern_count INTEGER DEFAULT 0,
    distressed_count INTEGER DEFAULT 0,
    severe_distress_count INTEGER DEFAULT 0,
    avg_z_score REAL,
    avg_m_score REAL,
    avg_f_score REAL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================================
-- SCHEDULER JOB RUNS (Monitoring)
-- ============================================================================

CREATE TABLE IF NOT EXISTS scheduler_job_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    duration_seconds REAL,
    records_processed INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    success INTEGER DEFAULT 1,
    error_count INTEGER DEFAULT 0,
    errors TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_job_runs_name_time ON scheduler_job_runs(job_name, start_time DESC);

-- Insert version
INSERT OR IGNORE INTO schema_version (version, description)
VALUES ('7.0.4', 'RL tables - valuation_outcomes, training, credit_risk');
