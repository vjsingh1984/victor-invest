-- RL System Database Schema
-- Supports outcome tracking, reward calculation, and model performance monitoring
-- Part of Phase 2: Reinforcement Learning Integration

-- =============================================================================
-- VALUATION OUTCOMES TABLE
-- Stores predictions and actual outcomes for RL training
-- =============================================================================

CREATE TABLE IF NOT EXISTS valuation_outcomes (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    analysis_date DATE NOT NULL,
    fiscal_period VARCHAR(10),

    -- Prediction Data (captured at analysis time)
    blended_fair_value NUMERIC(12, 2),
    current_price NUMERIC(12, 2),
    predicted_upside_pct NUMERIC(8, 2),

    -- Individual Model Fair Values
    dcf_fair_value NUMERIC(12, 2),
    pe_fair_value NUMERIC(12, 2),
    ps_fair_value NUMERIC(12, 2),
    evebitda_fair_value NUMERIC(12, 2),
    pb_fair_value NUMERIC(12, 2),
    ggm_fair_value NUMERIC(12, 2),

    -- Weights Used (for understanding what was predicted)
    model_weights JSONB,  -- {"dcf": 50, "pe": 30, "ps": 20, ...}
    tier_classification VARCHAR(50),

    -- Context Features (for RL state reconstruction)
    -- Stored as JSONB for flexibility and future feature additions
    context_features JSONB,
    -- Example: {
    --   "sector": "Technology",
    --   "industry": "Software - Application",
    --   "growth_stage": "high_growth",
    --   "company_size": "large_cap",
    --   "profitability_score": 0.75,
    --   "pe_level": 0.6,
    --   "revenue_growth": 0.25,
    --   "fcf_margin": 0.15,
    --   "rule_of_40_score": 45,
    --   "data_quality_score": 85,
    --   "technical_trend": 0.3,
    --   "market_sentiment": 0.2,
    --   ...
    -- }

    -- Entry/Exit Dates for position tracking
    entry_date DATE,  -- Date when position was entered (same as analysis_date for backtests)
    exit_date_30d DATE,  -- Date 30 days after entry
    exit_date_90d DATE,  -- Date 90 days after entry
    exit_date_365d DATE,  -- Date 365 days after entry

    -- Outcome Data (updated after 30/90/365 days)
    actual_price_30d NUMERIC(12, 2),
    actual_price_90d NUMERIC(12, 2),
    actual_price_365d NUMERIC(12, 2),
    outcome_updated_at TIMESTAMP,

    -- Calculated Rewards (updated after outcomes known)
    reward_30d NUMERIC(5, 3),  -- -1.0 to 1.0
    reward_90d NUMERIC(5, 3),  -- -1.0 to 1.0
    reward_365d NUMERIC(5, 3), -- -1.0 to 1.0

    -- Per-Model Rewards (for model-specific learning)
    per_model_rewards JSONB,
    -- Example: {
    --   "dcf": {"reward_30d": 0.8, "reward_90d": 0.7, "error_pct": 12.5},
    --   "pe": {"reward_30d": 0.3, "reward_90d": 0.4, "error_pct": 25.0},
    --   ...
    -- }

    -- RL Training Metadata
    used_for_training BOOLEAN DEFAULT FALSE,
    training_batch_id INTEGER,

    -- A/B Test Tracking
    ab_test_group VARCHAR(20),  -- 'rl', 'baseline', 'control'
    policy_version VARCHAR(50),

    -- Position type for dual policy signals
    position_type VARCHAR(20) DEFAULT 'inferred',  -- 'LONG', 'SHORT', 'inferred'

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure uniqueness: one prediction per symbol per day per position type
    UNIQUE(symbol, analysis_date, position_type)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_outcomes_symbol ON valuation_outcomes(symbol);
CREATE INDEX IF NOT EXISTS idx_outcomes_date ON valuation_outcomes(analysis_date);
CREATE INDEX IF NOT EXISTS idx_outcomes_sector ON valuation_outcomes((context_features->>'sector'));
CREATE INDEX IF NOT EXISTS idx_outcomes_tier ON valuation_outcomes(tier_classification);
CREATE INDEX IF NOT EXISTS idx_outcomes_ab_test ON valuation_outcomes(ab_test_group);
CREATE INDEX IF NOT EXISTS idx_outcomes_training ON valuation_outcomes(used_for_training) WHERE NOT used_for_training;
CREATE INDEX IF NOT EXISTS idx_outcomes_pending_update ON valuation_outcomes(analysis_date)
    WHERE actual_price_90d IS NULL;

-- =============================================================================
-- RL TRAINING BATCHES TABLE
-- Tracks training runs and model checkpoints
-- =============================================================================

CREATE TABLE IF NOT EXISTS rl_training_batches (
    id SERIAL PRIMARY KEY,
    batch_date TIMESTAMP NOT NULL,

    -- Training Configuration
    policy_type VARCHAR(50) NOT NULL,  -- 'contextual_bandit', 'dqn', 'hybrid'
    config_snapshot JSONB,  -- Full config used for training

    -- Training Data Stats
    num_experiences INTEGER,
    train_size INTEGER,
    validation_size INTEGER,
    test_size INTEGER,

    -- Training Metrics
    train_loss NUMERIC(10, 6),
    validation_loss NUMERIC(10, 6),
    train_reward_mean NUMERIC(5, 3),
    validation_reward_mean NUMERIC(5, 3),

    -- Model Artifact
    model_path VARCHAR(255),
    model_version VARCHAR(50),

    -- Evaluation Metrics (vs baseline)
    baseline_mape NUMERIC(8, 4),  -- Mean Absolute Percentage Error
    rl_mape NUMERIC(8, 4),
    mape_improvement_pct NUMERIC(8, 4),
    direction_accuracy NUMERIC(5, 3),  -- % correct buy/sell

    -- Status
    status VARCHAR(20) DEFAULT 'completed',  -- 'running', 'completed', 'failed'
    error_message TEXT,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_batches_date ON rl_training_batches(batch_date);
CREATE INDEX IF NOT EXISTS idx_training_batches_policy ON rl_training_batches(policy_type);

-- =============================================================================
-- RL POLICY METRICS TABLE
-- Tracks policy performance over time by sector/tier
-- =============================================================================

CREATE TABLE IF NOT EXISTS rl_policy_metrics (
    id SERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,

    -- Grouping Dimensions
    sector VARCHAR(100),
    industry VARCHAR(100),
    tier_classification VARCHAR(50),

    -- Aggregated Performance Metrics
    num_predictions INTEGER,
    avg_reward_30d NUMERIC(5, 3),
    avg_reward_90d NUMERIC(5, 3),
    avg_error_pct NUMERIC(8, 2),
    direction_accuracy NUMERIC(5, 3),

    -- Model-Specific Metrics
    model_performance JSONB,
    -- Example: {
    --   "dcf": {"avg_error": 15.2, "direction_accuracy": 0.65},
    --   "pe": {"avg_error": 22.5, "direction_accuracy": 0.55},
    --   ...
    -- }

    -- A/B Test Results
    rl_avg_reward NUMERIC(5, 3),
    baseline_avg_reward NUMERIC(5, 3),
    statistical_significance NUMERIC(5, 3),  -- p-value

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(metric_date, sector, industry, tier_classification)
);

CREATE INDEX IF NOT EXISTS idx_policy_metrics_date ON rl_policy_metrics(metric_date);
CREATE INDEX IF NOT EXISTS idx_policy_metrics_sector ON rl_policy_metrics(sector);

-- =============================================================================
-- TRIGGER: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_valuation_outcomes_updated_at ON valuation_outcomes;
CREATE TRIGGER update_valuation_outcomes_updated_at
    BEFORE UPDATE ON valuation_outcomes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- HELPER VIEWS
-- =============================================================================

-- View: Predictions ready for outcome update (30+ days old, no 30d price yet)
CREATE OR REPLACE VIEW v_pending_30d_updates AS
SELECT
    id, symbol, analysis_date,
    blended_fair_value, current_price,
    CURRENT_DATE - analysis_date AS days_since_analysis
FROM valuation_outcomes
WHERE actual_price_30d IS NULL
  AND analysis_date <= CURRENT_DATE - INTERVAL '30 days';

-- View: Predictions ready for outcome update (90+ days old, no 90d price yet)
CREATE OR REPLACE VIEW v_pending_90d_updates AS
SELECT
    id, symbol, analysis_date,
    blended_fair_value, current_price,
    CURRENT_DATE - analysis_date AS days_since_analysis
FROM valuation_outcomes
WHERE actual_price_90d IS NULL
  AND analysis_date <= CURRENT_DATE - INTERVAL '90 days';

-- View: Complete experiences for training (have 90d outcomes)
CREATE OR REPLACE VIEW v_training_ready_experiences AS
SELECT
    id, symbol, analysis_date,
    blended_fair_value, current_price,
    model_weights, tier_classification, context_features,
    actual_price_30d, actual_price_90d,
    reward_30d, reward_90d
FROM valuation_outcomes
WHERE reward_90d IS NOT NULL
  AND used_for_training = FALSE;

-- View: Policy performance summary by sector
CREATE OR REPLACE VIEW v_sector_performance AS
SELECT
    context_features->>'sector' AS sector,
    COUNT(*) AS num_predictions,
    AVG(reward_90d) AS avg_reward,
    AVG(ABS(blended_fair_value - actual_price_90d) / actual_price_90d * 100) AS avg_error_pct,
    SUM(CASE WHEN
        (blended_fair_value > current_price AND actual_price_90d > current_price) OR
        (blended_fair_value < current_price AND actual_price_90d < current_price)
        THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS direction_accuracy
FROM valuation_outcomes
WHERE reward_90d IS NOT NULL
GROUP BY context_features->>'sector'
ORDER BY avg_reward DESC;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE valuation_outcomes IS 'Stores valuation predictions and actual outcomes for RL training';
COMMENT ON COLUMN valuation_outcomes.context_features IS 'JSONB storing RL state features for training';
COMMENT ON COLUMN valuation_outcomes.per_model_rewards IS 'Individual model accuracy for model-specific weight learning';
COMMENT ON COLUMN valuation_outcomes.ab_test_group IS 'A/B test assignment: rl, baseline, or control';

COMMENT ON TABLE rl_training_batches IS 'Tracks RL model training runs and checkpoints';
COMMENT ON TABLE rl_policy_metrics IS 'Aggregated policy performance metrics by segment';
