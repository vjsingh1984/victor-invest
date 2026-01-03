-- Migration: Add Incremental Tracking Columns
-- Version: 002
-- Description: Add source tracking columns for incremental and idempotent data collection
-- Date: 2026-01-02
--
-- This migration adds:
-- 1. source_hash - SHA256 hash of record content for change detection
-- 2. source_fetch_timestamp - When record was fetched from external source
-- 3. last_verified_at - When record was last verified against source
-- 4. Indices for efficient incremental queries

BEGIN;

-- ================================================================================================
-- Treasury Yields - Natural key: date (already exists as PRIMARY KEY)
-- Add source tracking for change detection
-- ================================================================================================
ALTER TABLE treasury_yields
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS last_verified_at TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_treasury_yields_fetch_ts
    ON treasury_yields(source_fetch_timestamp DESC);

-- ================================================================================================
-- Macro Indicators - Natural key: series_id (already UNIQUE)
-- Indicators metadata rarely changes, values need tracking
-- ================================================================================================
ALTER TABLE macro_indicators
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS last_verified_at TIMESTAMP;

-- Macro Indicator Values - Natural key: (indicator_id, date) already PRIMARY KEY
ALTER TABLE macro_indicator_values
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_fetch_timestamp TIMESTAMP DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_macro_values_fetch_ts
    ON macro_indicator_values(source_fetch_timestamp DESC);

-- Track high watermark per indicator for efficient incremental fetching
CREATE TABLE IF NOT EXISTS macro_indicator_watermarks (
    indicator_id INTEGER PRIMARY KEY REFERENCES macro_indicators(id),
    last_observation_date DATE NOT NULL,
    last_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    fetch_count INTEGER DEFAULT 1
);

-- ================================================================================================
-- Form 4 Filings - Natural key: accession_number (SEC provides unique IDs)
-- Add missing columns based on collector code analysis
-- ================================================================================================

-- First, add any missing columns from the collector that the schema doesn't have
ALTER TABLE form4_filings
    ADD COLUMN IF NOT EXISTS cik VARCHAR(20),
    ADD COLUMN IF NOT EXISTS owner_name VARCHAR(200),
    ADD COLUMN IF NOT EXISTS owner_title VARCHAR(200),
    ADD COLUMN IF NOT EXISTS is_director BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS is_officer BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS transaction_code VARCHAR(10),
    ADD COLUMN IF NOT EXISTS is_significant BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS significance_reasons TEXT[],
    ADD COLUMN IF NOT EXISTS filing_data JSONB;

-- Add source tracking
ALTER TABLE form4_filings
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_fetch_timestamp TIMESTAMP DEFAULT NOW();

-- Create index on accession_number for incremental lookups
CREATE INDEX IF NOT EXISTS idx_form4_accession
    ON form4_filings(accession_number);

CREATE INDEX IF NOT EXISTS idx_form4_fetch_ts
    ON form4_filings(source_fetch_timestamp DESC);

-- Track last fetched accession per symbol for incremental SEC API queries
CREATE TABLE IF NOT EXISTS form4_fetch_watermarks (
    symbol VARCHAR(10) PRIMARY KEY,
    last_accession_number VARCHAR(30),
    last_filing_date DATE,
    last_fetch_timestamp TIMESTAMP DEFAULT NOW()
);

-- ================================================================================================
-- Insider Sentiment - Derived table (calculated from form4_filings)
-- Add tracking for recalculation decisions
-- ================================================================================================
ALTER TABLE insider_sentiment
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS last_recalculated_at TIMESTAMP DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS source_filing_count INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_insider_sentiment_recalc
    ON insider_sentiment(last_recalculated_at DESC);

-- ================================================================================================
-- Form 13F Filings - Natural key: accession_number (SEC provides unique IDs)
-- ================================================================================================
ALTER TABLE institutions
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_fetch_timestamp TIMESTAMP DEFAULT NOW();

ALTER TABLE form13f_filings
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_fetch_timestamp TIMESTAMP DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_form13f_filings_accession
    ON form13f_filings(accession_number);

CREATE INDEX IF NOT EXISTS idx_form13f_filings_fetch_ts
    ON form13f_filings(source_fetch_timestamp DESC);

ALTER TABLE form13f_holdings
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_fetch_timestamp TIMESTAMP DEFAULT NOW();

-- Track last fetched 13F per institution for incremental processing
CREATE TABLE IF NOT EXISTS form13f_fetch_watermarks (
    institution_cik VARCHAR(20) PRIMARY KEY,
    last_accession_number VARCHAR(30),
    last_report_quarter DATE,
    last_fetch_timestamp TIMESTAMP DEFAULT NOW()
);

-- ================================================================================================
-- Institutional Ownership - Derived table (aggregated from form13f_holdings)
-- ================================================================================================
ALTER TABLE institutional_ownership
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS last_recalculated_at TIMESTAMP DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS source_filing_count INTEGER DEFAULT 0;

-- ================================================================================================
-- Short Interest - Natural key: (symbol, settlement_date) already PRIMARY KEY
-- FINRA releases on specific settlement dates
-- ================================================================================================
ALTER TABLE short_interest
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_fetch_timestamp TIMESTAMP DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_short_interest_fetch_ts
    ON short_interest(source_fetch_timestamp DESC);

-- Track last settlement date per symbol for incremental fetching
CREATE TABLE IF NOT EXISTS short_interest_watermarks (
    symbol VARCHAR(10) PRIMARY KEY,
    last_settlement_date DATE,
    last_fetch_timestamp TIMESTAMP DEFAULT NOW()
);

-- ================================================================================================
-- Market Regime History - Natural key: snapshot_date (already UNIQUE)
-- ================================================================================================
ALTER TABLE market_regime_history
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_fetch_timestamp TIMESTAMP DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_market_regime_fetch_ts
    ON market_regime_history(source_fetch_timestamp DESC);

-- ================================================================================================
-- Credit Risk Scores - Calculated from financial data
-- ================================================================================================
ALTER TABLE credit_risk_scores
    ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_fetch_timestamp TIMESTAMP DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS input_data_hash VARCHAR(64);

CREATE INDEX IF NOT EXISTS idx_credit_risk_fetch_ts
    ON credit_risk_scores(source_fetch_timestamp DESC);

-- ================================================================================================
-- Scheduler Job Runs - Add high watermark tracking
-- ================================================================================================
ALTER TABLE scheduler_job_runs
    ADD COLUMN IF NOT EXISTS high_watermark_date DATE,
    ADD COLUMN IF NOT EXISTS high_watermark_value VARCHAR(100),
    ADD COLUMN IF NOT EXISTS records_skipped INTEGER DEFAULT 0;

-- ================================================================================================
-- Create function to compute content hash
-- ================================================================================================
CREATE OR REPLACE FUNCTION compute_record_hash(content TEXT)
RETURNS VARCHAR(64) AS $$
BEGIN
    RETURN encode(digest(content, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ================================================================================================
-- Create view for incremental fetch status
-- ================================================================================================
CREATE OR REPLACE VIEW incremental_fetch_status AS
SELECT
    'treasury_yields' as table_name,
    MAX(date) as last_record_date,
    MAX(source_fetch_timestamp) as last_fetch_timestamp,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE source_hash IS NOT NULL) as records_with_hash
FROM treasury_yields
UNION ALL
SELECT
    'macro_indicator_values',
    MAX(date),
    MAX(source_fetch_timestamp),
    COUNT(*),
    COUNT(*) FILTER (WHERE source_hash IS NOT NULL)
FROM macro_indicator_values
UNION ALL
SELECT
    'form4_filings',
    MAX(filing_date),
    MAX(source_fetch_timestamp),
    COUNT(*),
    COUNT(*) FILTER (WHERE source_hash IS NOT NULL)
FROM form4_filings
UNION ALL
SELECT
    'form13f_filings',
    MAX(filing_date),
    MAX(source_fetch_timestamp),
    COUNT(*),
    COUNT(*) FILTER (WHERE source_hash IS NOT NULL)
FROM form13f_filings
UNION ALL
SELECT
    'short_interest',
    MAX(settlement_date),
    MAX(source_fetch_timestamp),
    COUNT(*),
    COUNT(*) FILTER (WHERE source_hash IS NOT NULL)
FROM short_interest
UNION ALL
SELECT
    'market_regime_history',
    MAX(snapshot_date),
    MAX(source_fetch_timestamp),
    COUNT(*),
    COUNT(*) FILTER (WHERE source_hash IS NOT NULL)
FROM market_regime_history
UNION ALL
SELECT
    'credit_risk_scores',
    MAX(calculation_date),
    MAX(source_fetch_timestamp),
    COUNT(*),
    COUNT(*) FILTER (WHERE source_hash IS NOT NULL)
FROM credit_risk_scores;

-- ================================================================================================
-- Record schema version
-- ================================================================================================
INSERT INTO schema_version (version, description)
VALUES ('6.2.0', 'Added incremental tracking columns for idempotent data collection')
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- Show status
SELECT '002_add_incremental_tracking.sql applied successfully' AS status;
