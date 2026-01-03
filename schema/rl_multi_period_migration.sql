-- RL Multi-Period Data Migration & View
-- Consolidates multi-period data into JSONB (per_model_rewards) as source of truth
-- Creates view for normalized column access
--
-- Run this migration to:
-- 1. Populate JSONB multi_period data from existing columns
-- 2. Create view for easy column-based querying

-- =============================================================================
-- STEP 1: MIGRATE EXISTING COLUMN DATA INTO JSONB
-- =============================================================================

-- Update existing records to include multi_period data in per_model_rewards JSONB
-- This preserves existing data while making JSONB the source of truth

UPDATE valuation_outcomes
SET per_model_rewards = COALESCE(per_model_rewards, '{}'::jsonb) || jsonb_build_object(
    'multi_period', jsonb_build_object(
        'entry_date', entry_date::text,
        'prices', jsonb_build_object(
            '1m', actual_price_30d,
            '3m', actual_price_90d,
            '12m', actual_price_365d
        ),
        'exit_dates', jsonb_build_object(
            '1m', CASE WHEN actual_price_30d IS NOT NULL THEN (entry_date + INTERVAL '30 days')::date::text END,
            '3m', CASE WHEN actual_price_90d IS NOT NULL THEN (entry_date + INTERVAL '90 days')::date::text END,
            '12m', CASE WHEN actual_price_365d IS NOT NULL THEN (entry_date + INTERVAL '365 days')::date::text END
        ),
        'long_rewards', CASE
            WHEN position_type = 'LONG' THEN jsonb_build_object(
                '1m', reward_30d,
                '3m', reward_90d,
                '12m', reward_365d
            )
            ELSE '{}'::jsonb
        END,
        'short_rewards', CASE
            WHEN position_type = 'SHORT' THEN jsonb_build_object(
                '1m', reward_30d,
                '3m', reward_90d,
                '12m', reward_365d
            )
            ELSE '{}'::jsonb
        END
    )
)
WHERE per_model_rewards IS NULL
   OR per_model_rewards->'multi_period' IS NULL
   OR per_model_rewards->'multi_period'->'entry_date' IS NULL;

-- =============================================================================
-- STEP 2: CREATE NORMALIZED VIEW FOR COLUMN ACCESS
-- =============================================================================

-- Drop existing view if it exists
DROP VIEW IF EXISTS v_valuation_multi_period CASCADE;

-- Create view that extracts JSONB multi_period data as columns
-- This provides backward-compatible column access while JSONB is source of truth
CREATE OR REPLACE VIEW v_valuation_multi_period AS
SELECT
    -- Core identification
    id,
    symbol,
    analysis_date,
    position_type,

    -- Valuation data
    blended_fair_value,
    current_price,
    predicted_upside_pct,
    tier_classification,

    -- Individual model fair values
    dcf_fair_value,
    pe_fair_value,
    ps_fair_value,
    evebitda_fair_value,
    pb_fair_value,
    ggm_fair_value,

    -- Entry date (from column, as it's the base reference)
    COALESCE(
        entry_date,
        (per_model_rewards->'multi_period'->>'entry_date')::date
    ) AS entry_date,

    -- Multi-period prices (from JSONB)
    (per_model_rewards->'multi_period'->'prices'->>'1m')::numeric AS price_1m,
    (per_model_rewards->'multi_period'->'prices'->>'3m')::numeric AS price_3m,
    (per_model_rewards->'multi_period'->'prices'->>'6m')::numeric AS price_6m,
    (per_model_rewards->'multi_period'->'prices'->>'12m')::numeric AS price_12m,
    (per_model_rewards->'multi_period'->'prices'->>'18m')::numeric AS price_18m,
    (per_model_rewards->'multi_period'->'prices'->>'24m')::numeric AS price_24m,
    (per_model_rewards->'multi_period'->'prices'->>'36m')::numeric AS price_36m,

    -- Multi-period exit dates (from JSONB)
    (per_model_rewards->'multi_period'->'exit_dates'->>'1m')::date AS exit_date_1m,
    (per_model_rewards->'multi_period'->'exit_dates'->>'3m')::date AS exit_date_3m,
    (per_model_rewards->'multi_period'->'exit_dates'->>'6m')::date AS exit_date_6m,
    (per_model_rewards->'multi_period'->'exit_dates'->>'12m')::date AS exit_date_12m,
    (per_model_rewards->'multi_period'->'exit_dates'->>'18m')::date AS exit_date_18m,
    (per_model_rewards->'multi_period'->'exit_dates'->>'24m')::date AS exit_date_24m,
    (per_model_rewards->'multi_period'->'exit_dates'->>'36m')::date AS exit_date_36m,

    -- Long rewards (from JSONB)
    (per_model_rewards->'multi_period'->'long_rewards'->>'1m')::numeric AS long_reward_1m,
    (per_model_rewards->'multi_period'->'long_rewards'->>'3m')::numeric AS long_reward_3m,
    (per_model_rewards->'multi_period'->'long_rewards'->>'6m')::numeric AS long_reward_6m,
    (per_model_rewards->'multi_period'->'long_rewards'->>'12m')::numeric AS long_reward_12m,
    (per_model_rewards->'multi_period'->'long_rewards'->>'18m')::numeric AS long_reward_18m,
    (per_model_rewards->'multi_period'->'long_rewards'->>'24m')::numeric AS long_reward_24m,
    (per_model_rewards->'multi_period'->'long_rewards'->>'36m')::numeric AS long_reward_36m,

    -- Short rewards (from JSONB)
    (per_model_rewards->'multi_period'->'short_rewards'->>'1m')::numeric AS short_reward_1m,
    (per_model_rewards->'multi_period'->'short_rewards'->>'3m')::numeric AS short_reward_3m,
    (per_model_rewards->'multi_period'->'short_rewards'->>'6m')::numeric AS short_reward_6m,
    (per_model_rewards->'multi_period'->'short_rewards'->>'12m')::numeric AS short_reward_12m,
    (per_model_rewards->'multi_period'->'short_rewards'->>'18m')::numeric AS short_reward_18m,
    (per_model_rewards->'multi_period'->'short_rewards'->>'24m')::numeric AS short_reward_24m,
    (per_model_rewards->'multi_period'->'short_rewards'->>'36m')::numeric AS short_reward_36m,

    -- Position-specific reward (based on position_type)
    CASE position_type
        WHEN 'LONG' THEN (per_model_rewards->'multi_period'->'long_rewards'->>'1m')::numeric
        WHEN 'SHORT' THEN (per_model_rewards->'multi_period'->'short_rewards'->>'1m')::numeric
    END AS reward_1m,
    CASE position_type
        WHEN 'LONG' THEN (per_model_rewards->'multi_period'->'long_rewards'->>'3m')::numeric
        WHEN 'SHORT' THEN (per_model_rewards->'multi_period'->'short_rewards'->>'3m')::numeric
    END AS reward_3m,
    CASE position_type
        WHEN 'LONG' THEN (per_model_rewards->'multi_period'->'long_rewards'->>'6m')::numeric
        WHEN 'SHORT' THEN (per_model_rewards->'multi_period'->'short_rewards'->>'6m')::numeric
    END AS reward_6m,
    CASE position_type
        WHEN 'LONG' THEN (per_model_rewards->'multi_period'->'long_rewards'->>'12m')::numeric
        WHEN 'SHORT' THEN (per_model_rewards->'multi_period'->'short_rewards'->>'12m')::numeric
    END AS reward_12m,

    -- Context features
    context_features,

    -- Metadata
    created_at,
    updated_at

FROM valuation_outcomes;

-- =============================================================================
-- STEP 3: CREATE SUMMARY VIEW FOR PERFORMANCE ANALYSIS
-- =============================================================================

DROP VIEW IF EXISTS v_holding_period_performance CASCADE;

CREATE OR REPLACE VIEW v_holding_period_performance AS
SELECT
    symbol,
    position_type,
    COUNT(*) AS num_predictions,

    -- Average rewards by holding period
    ROUND(AVG((per_model_rewards->'multi_period'->'long_rewards'->>'1m')::numeric), 4) AS avg_long_1m,
    ROUND(AVG((per_model_rewards->'multi_period'->'long_rewards'->>'3m')::numeric), 4) AS avg_long_3m,
    ROUND(AVG((per_model_rewards->'multi_period'->'long_rewards'->>'6m')::numeric), 4) AS avg_long_6m,
    ROUND(AVG((per_model_rewards->'multi_period'->'long_rewards'->>'12m')::numeric), 4) AS avg_long_12m,
    ROUND(AVG((per_model_rewards->'multi_period'->'long_rewards'->>'18m')::numeric), 4) AS avg_long_18m,
    ROUND(AVG((per_model_rewards->'multi_period'->'long_rewards'->>'24m')::numeric), 4) AS avg_long_24m,
    ROUND(AVG((per_model_rewards->'multi_period'->'long_rewards'->>'36m')::numeric), 4) AS avg_long_36m,

    ROUND(AVG((per_model_rewards->'multi_period'->'short_rewards'->>'1m')::numeric), 4) AS avg_short_1m,
    ROUND(AVG((per_model_rewards->'multi_period'->'short_rewards'->>'3m')::numeric), 4) AS avg_short_3m,
    ROUND(AVG((per_model_rewards->'multi_period'->'short_rewards'->>'6m')::numeric), 4) AS avg_short_6m,
    ROUND(AVG((per_model_rewards->'multi_period'->'short_rewards'->>'12m')::numeric), 4) AS avg_short_12m,

    -- Best holding period identification
    MIN(analysis_date) AS earliest_prediction,
    MAX(analysis_date) AS latest_prediction

FROM valuation_outcomes
WHERE per_model_rewards->'multi_period' IS NOT NULL
GROUP BY symbol, position_type;

-- =============================================================================
-- STEP 4: CREATE GIN INDEX FOR JSONB QUERIES
-- =============================================================================

-- Create GIN index on per_model_rewards for efficient JSONB queries
CREATE INDEX IF NOT EXISTS idx_valuation_outcomes_per_model_rewards_gin
ON valuation_outcomes USING GIN (per_model_rewards);

-- Create index on multi_period path for common queries
CREATE INDEX IF NOT EXISTS idx_valuation_outcomes_multi_period
ON valuation_outcomes USING GIN ((per_model_rewards->'multi_period'));

-- =============================================================================
-- DOCUMENTATION
-- =============================================================================

COMMENT ON VIEW v_valuation_multi_period IS
'Normalized view of valuation_outcomes with multi-period data extracted from JSONB.
Source of truth: per_model_rewards->multi_period JSONB field.
Provides column access for: prices, exit_dates, long_rewards, short_rewards
Holding periods: 1m (30d), 3m (90d), 6m (180d), 12m (365d), 18m, 24m, 36m';

COMMENT ON VIEW v_holding_period_performance IS
'Aggregated performance metrics by symbol and position type across all holding periods.
Use for comparing which holding periods perform best for each symbol.';

-- =============================================================================
-- VERIFICATION QUERIES (run after migration)
-- =============================================================================

-- Check migration success
-- SELECT COUNT(*) AS total,
--        COUNT(per_model_rewards->'multi_period') AS has_multi_period
-- FROM valuation_outcomes;

-- Sample the view
-- SELECT * FROM v_valuation_multi_period LIMIT 5;

-- Check performance by holding period
-- SELECT * FROM v_holding_period_performance WHERE symbol = 'AAPL';
