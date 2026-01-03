-- Migration: Add ML/RL Training Datasets
-- Version: 003
-- Description: Add tables for analyst estimates, news sentiment, earnings quality,
--              dividend/buyback history, and Fama-French factors
-- Date: 2026-01-02

BEGIN;

-- ================================================================================================
-- 1. ANALYST ESTIMATES
-- Source: Finnhub API (free tier)
-- Natural key: (symbol, period_type, period_end_date)
-- ================================================================================================
CREATE TABLE IF NOT EXISTS analyst_estimates (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    period_type VARCHAR(10) NOT NULL,  -- 'quarterly', 'annual'
    period_end_date DATE NOT NULL,     -- End date of the period

    -- EPS Estimates
    eps_estimate_avg DECIMAL(12, 4),
    eps_estimate_high DECIMAL(12, 4),
    eps_estimate_low DECIMAL(12, 4),
    eps_estimate_count INTEGER,
    eps_actual DECIMAL(12, 4),
    eps_surprise DECIMAL(12, 4),
    eps_surprise_pct DECIMAL(10, 4),

    -- Revenue Estimates
    revenue_estimate_avg DECIMAL(20, 2),
    revenue_estimate_high DECIMAL(20, 2),
    revenue_estimate_low DECIMAL(20, 2),
    revenue_estimate_count INTEGER,
    revenue_actual DECIMAL(20, 2),
    revenue_surprise DECIMAL(20, 2),
    revenue_surprise_pct DECIMAL(10, 4),

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (symbol, period_type, period_end_date)
);

CREATE INDEX IF NOT EXISTS idx_analyst_estimates_symbol ON analyst_estimates(symbol);
CREATE INDEX IF NOT EXISTS idx_analyst_estimates_date ON analyst_estimates(period_end_date DESC);
CREATE INDEX IF NOT EXISTS idx_analyst_estimates_fetch_ts ON analyst_estimates(source_fetch_timestamp DESC);

-- Analyst Price Targets
CREATE TABLE IF NOT EXISTS analyst_price_targets (
    symbol VARCHAR(10) PRIMARY KEY,
    target_high DECIMAL(12, 2),
    target_low DECIMAL(12, 2),
    target_mean DECIMAL(12, 2),
    target_median DECIMAL(12, 2),
    analyst_count INTEGER,
    last_updated DATE,

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_price_targets_fetch_ts ON analyst_price_targets(source_fetch_timestamp DESC);

-- Analyst Recommendations
CREATE TABLE IF NOT EXISTS analyst_recommendations (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    period_date DATE NOT NULL,
    strong_buy INTEGER DEFAULT 0,
    buy INTEGER DEFAULT 0,
    hold INTEGER DEFAULT 0,
    sell INTEGER DEFAULT 0,
    strong_sell INTEGER DEFAULT 0,

    -- Calculated scores
    consensus_score DECIMAL(4, 2),  -- Weighted average 1-5
    total_analysts INTEGER,

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (symbol, period_date)
);

CREATE INDEX IF NOT EXISTS idx_analyst_recs_symbol ON analyst_recommendations(symbol);
CREATE INDEX IF NOT EXISTS idx_analyst_recs_date ON analyst_recommendations(period_date DESC);

-- Estimate Revisions (track momentum)
CREATE TABLE IF NOT EXISTS estimate_revisions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    revision_date DATE NOT NULL,

    -- 7-day revisions
    eps_up_7d INTEGER DEFAULT 0,
    eps_down_7d INTEGER DEFAULT 0,
    rev_up_7d INTEGER DEFAULT 0,
    rev_down_7d INTEGER DEFAULT 0,

    -- 30-day revisions
    eps_up_30d INTEGER DEFAULT 0,
    eps_down_30d INTEGER DEFAULT 0,
    rev_up_30d INTEGER DEFAULT 0,
    rev_down_30d INTEGER DEFAULT 0,

    -- 90-day revisions
    eps_up_90d INTEGER DEFAULT 0,
    eps_down_90d INTEGER DEFAULT 0,
    rev_up_90d INTEGER DEFAULT 0,
    rev_down_90d INTEGER DEFAULT 0,

    -- Calculated momentum scores
    eps_revision_momentum DECIMAL(6, 4),  -- (up - down) / total
    rev_revision_momentum DECIMAL(6, 4),

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (symbol, revision_date)
);

CREATE INDEX IF NOT EXISTS idx_revisions_symbol ON estimate_revisions(symbol);
CREATE INDEX IF NOT EXISTS idx_revisions_date ON estimate_revisions(revision_date DESC);

-- ================================================================================================
-- 2. NEWS SENTIMENT
-- Source: Finnhub News Sentiment API (free tier)
-- Natural key: (symbol, sentiment_date)
-- ================================================================================================
CREATE TABLE IF NOT EXISTS news_sentiment (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    sentiment_date DATE NOT NULL,

    -- Article counts by sentiment
    articles_in_period INTEGER DEFAULT 0,
    positive_count INTEGER DEFAULT 0,
    negative_count INTEGER DEFAULT 0,
    neutral_count INTEGER DEFAULT 0,

    -- Sentiment scores (-1 to +1)
    sentiment_score DECIMAL(6, 4),       -- Average sentiment
    sentiment_std_dev DECIMAL(6, 4),     -- Volatility of sentiment
    buzz_score DECIMAL(10, 4),           -- Relative volume of news

    -- Company-specific metrics (from Finnhub)
    company_news_score DECIMAL(6, 4),
    sector_avg_sentiment DECIMAL(6, 4),
    sector_avg_news_volume DECIMAL(10, 4),

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (symbol, sentiment_date)
);

CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol ON news_sentiment(symbol);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_date ON news_sentiment(sentiment_date DESC);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_fetch_ts ON news_sentiment(source_fetch_timestamp DESC);

-- News Sentiment Watermarks
CREATE TABLE IF NOT EXISTS news_sentiment_watermarks (
    symbol VARCHAR(10) PRIMARY KEY,
    last_sentiment_date DATE,
    last_fetch_timestamp TIMESTAMP DEFAULT NOW()
);

-- ================================================================================================
-- 3. EARNINGS QUALITY METRICS
-- Source: Calculated from SEC data (no external API needed)
-- Natural key: (symbol, fiscal_year, fiscal_period)
-- ================================================================================================
CREATE TABLE IF NOT EXISTS earnings_quality (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_period VARCHAR(10) NOT NULL,  -- 'Q1', 'Q2', 'Q3', 'Q4', 'FY'

    -- Accrual Metrics
    total_accruals DECIMAL(20, 2),       -- Net Income - Operating Cash Flow
    accrual_ratio DECIMAL(10, 6),        -- Total Accruals / Avg Total Assets
    sloan_accruals DECIMAL(10, 6),       -- (WC Change - Depreciation) / Avg Assets

    -- Cash Conversion Quality
    cash_conversion_ratio DECIMAL(10, 4),  -- Operating Cash Flow / Net Income
    fcf_to_net_income DECIMAL(10, 4),      -- Free Cash Flow / Net Income

    -- Quality Scores (higher = better)
    piotroski_f_score INTEGER,           -- 0-9 (already exists but adding here for completeness)
    beneish_m_score DECIMAL(10, 4),      -- Manipulation score (< -2.22 is good)
    altman_z_score DECIMAL(10, 4),       -- Bankruptcy risk (> 2.99 is safe)

    -- Revenue Quality
    dso_change DECIMAL(10, 4),           -- Change in Days Sales Outstanding
    deferred_revenue_change DECIMAL(10, 4),

    -- Composite Score
    earnings_quality_score DECIMAL(6, 4),  -- 0-1 composite (higher = better)
    quality_flags TEXT[],                   -- Array of quality concerns

    -- Tracking
    input_data_hash VARCHAR(64),         -- Hash of input financial data
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (symbol, fiscal_year, fiscal_period)
);

CREATE INDEX IF NOT EXISTS idx_earnings_quality_symbol ON earnings_quality(symbol);
CREATE INDEX IF NOT EXISTS idx_earnings_quality_period ON earnings_quality(fiscal_year DESC, fiscal_period);

-- ================================================================================================
-- 4. DIVIDEND AND BUYBACK DATA
-- Source: Finnhub API (free tier) + SEC data
-- Natural key: (symbol, ex_dividend_date) for dividends, (symbol, quarter_date) for buybacks
-- ================================================================================================
CREATE TABLE IF NOT EXISTS dividend_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    ex_dividend_date DATE NOT NULL,
    payment_date DATE,
    record_date DATE,
    declaration_date DATE,

    -- Dividend details
    dividend_amount DECIMAL(12, 6),
    dividend_type VARCHAR(20),  -- 'regular', 'special', 'stock'
    frequency VARCHAR(20),      -- 'quarterly', 'monthly', 'annual', 'irregular'
    adjusted_amount DECIMAL(12, 6),  -- Split-adjusted

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (symbol, ex_dividend_date)
);

CREATE INDEX IF NOT EXISTS idx_dividend_symbol ON dividend_history(symbol);
CREATE INDEX IF NOT EXISTS idx_dividend_date ON dividend_history(ex_dividend_date DESC);

-- Buyback/Repurchase Data
CREATE TABLE IF NOT EXISTS buyback_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    quarter_date DATE NOT NULL,  -- End of quarter

    -- Repurchase amounts
    shares_repurchased NUMERIC(20, 0),
    repurchase_value DECIMAL(20, 2),
    avg_repurchase_price DECIMAL(12, 4),

    -- Authorization
    shares_authorized NUMERIC(20, 0),
    value_authorized DECIMAL(20, 2),
    authorization_remaining DECIMAL(20, 2),

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (symbol, quarter_date)
);

CREATE INDEX IF NOT EXISTS idx_buyback_symbol ON buyback_history(symbol);
CREATE INDEX IF NOT EXISTS idx_buyback_date ON buyback_history(quarter_date DESC);

-- Shareholder Yield Summary (aggregated view)
CREATE TABLE IF NOT EXISTS shareholder_yield (
    symbol VARCHAR(10) PRIMARY KEY,
    calculation_date DATE NOT NULL,

    -- TTM yields
    dividend_yield_ttm DECIMAL(10, 6),
    buyback_yield_ttm DECIMAL(10, 6),
    total_shareholder_yield DECIMAL(10, 6),

    -- Growth metrics
    dividend_growth_1y DECIMAL(10, 4),
    dividend_growth_3y DECIMAL(10, 4),
    dividend_growth_5y DECIMAL(10, 4),

    -- Streak tracking
    consecutive_dividend_years INTEGER DEFAULT 0,
    dividend_aristocrat BOOLEAN DEFAULT FALSE,  -- 25+ years
    dividend_king BOOLEAN DEFAULT FALSE,        -- 50+ years

    -- Payout sustainability
    payout_ratio DECIMAL(10, 4),
    fcf_payout_ratio DECIMAL(10, 4),

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================================================
-- 5. FAMA-FRENCH FACTORS
-- Source: Kenneth French Data Library (free)
-- Natural key: date
-- ================================================================================================
CREATE TABLE IF NOT EXISTS fama_french_factors (
    date DATE PRIMARY KEY,

    -- 5-factor model
    mkt_rf DECIMAL(12, 8),    -- Market minus risk-free
    smb DECIMAL(12, 8),       -- Small minus Big (size)
    hml DECIMAL(12, 8),       -- High minus Low (value)
    rmw DECIMAL(12, 8),       -- Robust minus Weak (profitability)
    cma DECIMAL(12, 8),       -- Conservative minus Aggressive (investment)
    rf DECIMAL(12, 8),        -- Risk-free rate

    -- Momentum factor
    umd DECIMAL(12, 8),       -- Up minus Down (momentum)

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ff_factors_date ON fama_french_factors(date DESC);
CREATE INDEX IF NOT EXISTS idx_ff_factors_fetch_ts ON fama_french_factors(source_fetch_timestamp DESC);

-- Stock Factor Exposures (rolling betas)
CREATE TABLE IF NOT EXISTS stock_factor_exposures (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    calculation_date DATE NOT NULL,
    lookback_days INTEGER NOT NULL,  -- e.g., 252 for 1 year

    -- Factor betas (exposures)
    beta_mkt DECIMAL(10, 6),
    beta_smb DECIMAL(10, 6),
    beta_hml DECIMAL(10, 6),
    beta_rmw DECIMAL(10, 6),
    beta_cma DECIMAL(10, 6),
    beta_umd DECIMAL(10, 6),

    -- Model fit
    r_squared DECIMAL(6, 4),
    alpha DECIMAL(12, 8),     -- Annualized alpha

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (symbol, calculation_date, lookback_days)
);

CREATE INDEX IF NOT EXISTS idx_factor_exposures_symbol ON stock_factor_exposures(symbol);
CREATE INDEX IF NOT EXISTS idx_factor_exposures_date ON stock_factor_exposures(calculation_date DESC);

-- ================================================================================================
-- 6. ENHANCED ECONOMIC INDICATORS (additional FRED series)
-- Extend existing macro_indicators with specific series for ML
-- ================================================================================================

-- Sector-Specific Leading Indicators
CREATE TABLE IF NOT EXISTS sector_indicators (
    id SERIAL PRIMARY KEY,
    sector VARCHAR(50) NOT NULL,
    indicator_date DATE NOT NULL,

    -- Housing (Real Estate, Construction, Home Improvement)
    housing_starts DECIMAL(20, 4),
    building_permits DECIMAL(20, 4),
    existing_home_sales DECIMAL(20, 4),
    mortgage_rate_30y DECIMAL(8, 4),

    -- Auto (Automotive, Auto Parts)
    auto_sales DECIMAL(20, 4),
    auto_inventory_days DECIMAL(10, 2),

    -- Technology
    semiconductor_shipments DECIMAL(20, 4),
    pce_tech_spending DECIMAL(20, 4),

    -- Consumer
    retail_sales DECIMAL(20, 4),
    consumer_confidence DECIMAL(10, 4),
    pce_total DECIMAL(20, 4),

    -- Industrial
    industrial_production DECIMAL(10, 4),
    capacity_utilization DECIMAL(10, 4),
    ism_manufacturing DECIMAL(10, 4),

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (sector, indicator_date)
);

CREATE INDEX IF NOT EXISTS idx_sector_indicators_date ON sector_indicators(indicator_date DESC);

-- ================================================================================================
-- 7. EARNINGS SURPRISE HISTORY
-- Track historical beat/miss patterns for each symbol
-- ================================================================================================
CREATE TABLE IF NOT EXISTS earnings_surprise_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    earnings_date DATE NOT NULL,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,

    -- EPS
    eps_estimate DECIMAL(12, 4),
    eps_actual DECIMAL(12, 4),
    eps_surprise DECIMAL(12, 4),
    eps_surprise_pct DECIMAL(10, 4),
    eps_beat BOOLEAN,

    -- Revenue
    revenue_estimate DECIMAL(20, 2),
    revenue_actual DECIMAL(20, 2),
    revenue_surprise DECIMAL(20, 2),
    revenue_surprise_pct DECIMAL(10, 4),
    revenue_beat BOOLEAN,

    -- Price reaction
    price_change_1d DECIMAL(10, 4),
    price_change_5d DECIMAL(10, 4),

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (symbol, earnings_date)
);

CREATE INDEX IF NOT EXISTS idx_earnings_surprise_symbol ON earnings_surprise_history(symbol);
CREATE INDEX IF NOT EXISTS idx_earnings_surprise_date ON earnings_surprise_history(earnings_date DESC);

-- Earnings Surprise Summary (current stats per symbol)
CREATE TABLE IF NOT EXISTS earnings_surprise_summary (
    symbol VARCHAR(10) PRIMARY KEY,
    calculation_date DATE NOT NULL,

    -- Beat rates (last 4, 8, 12 quarters)
    eps_beat_rate_4q DECIMAL(6, 4),
    eps_beat_rate_8q DECIMAL(6, 4),
    eps_beat_rate_12q DECIMAL(6, 4),

    revenue_beat_rate_4q DECIMAL(6, 4),
    revenue_beat_rate_8q DECIMAL(6, 4),
    revenue_beat_rate_12q DECIMAL(6, 4),

    -- Average surprise magnitude
    avg_eps_surprise_4q DECIMAL(10, 4),
    avg_eps_surprise_8q DECIMAL(10, 4),

    -- Consistency
    consecutive_eps_beats INTEGER DEFAULT 0,
    consecutive_rev_beats INTEGER DEFAULT 0,

    -- Tracking
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================================================
-- WATERMARK TABLES FOR INCREMENTAL FETCHING
-- ================================================================================================

CREATE TABLE IF NOT EXISTS analyst_estimates_watermarks (
    symbol VARCHAR(10) PRIMARY KEY,
    last_period_date DATE,
    last_fetch_timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dividend_watermarks (
    symbol VARCHAR(10) PRIMARY KEY,
    last_ex_date DATE,
    last_fetch_timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ff_factors_watermarks (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    last_date DATE,
    last_fetch_timestamp TIMESTAMP DEFAULT NOW()
);

-- ================================================================================================
-- VIEW: ML Feature Status
-- ================================================================================================
CREATE OR REPLACE VIEW ml_data_status AS
SELECT
    'analyst_estimates' as table_name,
    COUNT(DISTINCT symbol) as symbols_covered,
    MAX(period_end_date) as last_record_date,
    MAX(source_fetch_timestamp) as last_fetch_timestamp,
    COUNT(*) as total_records
FROM analyst_estimates
UNION ALL
SELECT 'analyst_price_targets', COUNT(*), MAX(last_updated), MAX(source_fetch_timestamp), COUNT(*)
FROM analyst_price_targets
UNION ALL
SELECT 'analyst_recommendations', COUNT(DISTINCT symbol), MAX(period_date), MAX(source_fetch_timestamp), COUNT(*)
FROM analyst_recommendations
UNION ALL
SELECT 'news_sentiment', COUNT(DISTINCT symbol), MAX(sentiment_date), MAX(source_fetch_timestamp), COUNT(*)
FROM news_sentiment
UNION ALL
SELECT 'earnings_quality', COUNT(DISTINCT symbol), NULL, MAX(source_fetch_timestamp), COUNT(*)
FROM earnings_quality
UNION ALL
SELECT 'dividend_history', COUNT(DISTINCT symbol), MAX(ex_dividend_date), MAX(source_fetch_timestamp), COUNT(*)
FROM dividend_history
UNION ALL
SELECT 'buyback_history', COUNT(DISTINCT symbol), MAX(quarter_date), MAX(source_fetch_timestamp), COUNT(*)
FROM buyback_history
UNION ALL
SELECT 'fama_french_factors', NULL, MAX(date), MAX(source_fetch_timestamp), COUNT(*)
FROM fama_french_factors
UNION ALL
SELECT 'stock_factor_exposures', COUNT(DISTINCT symbol), MAX(calculation_date), MAX(source_fetch_timestamp), COUNT(*)
FROM stock_factor_exposures
UNION ALL
SELECT 'earnings_surprise_history', COUNT(DISTINCT symbol), MAX(earnings_date), MAX(source_fetch_timestamp), COUNT(*)
FROM earnings_surprise_history;

-- ================================================================================================
-- Record schema version
-- ================================================================================================
INSERT INTO schema_version (version, description)
VALUES ('6.3.0', 'Added ML/RL training datasets: analyst estimates, news sentiment, earnings quality, dividends, factors')
ON CONFLICT (version) DO NOTHING;

COMMIT;

SELECT '003_add_ml_datasets.sql applied successfully' AS status;
