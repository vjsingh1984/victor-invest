-- Migration: Create 3-Table SEC Architecture (Raw/Processed/Metadata)
-- Date: 2025-11-02
-- Purpose: Separate raw SEC API data from processed quarterly data
-- Fixes: "0/8 ratios" bug caused by mixed raw/processed data

-- =============================================================================
-- TABLE 1: sec_companyfacts_raw (IMMUTABLE RAW SEC API RESPONSES)
-- =============================================================================

CREATE TABLE IF NOT EXISTS sec_companyfacts_raw (
    id                  SERIAL PRIMARY KEY,
    symbol              VARCHAR(10) NOT NULL,
    cik                 VARCHAR(10) NOT NULL,
    entity_name         VARCHAR(255),

    -- RAW SEC API RESPONSE (exact copy, never modify)
    companyfacts        JSONB NOT NULL,

    -- API METADATA
    api_version         VARCHAR(50) DEFAULT 'v1.0',
    fetched_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    api_response_size   INTEGER,
    api_checksum        VARCHAR(64),

    -- CONSTRAINTS
    CONSTRAINT sec_companyfacts_raw_symbol_unique UNIQUE (symbol),
    CONSTRAINT sec_companyfacts_raw_cik_unique UNIQUE (cik),
    CONSTRAINT sec_companyfacts_raw_us_gaap_check
        CHECK (companyfacts ? 'facts' AND companyfacts->'facts' ? 'us-gaap')
);

-- INDEXES for fast lookups
CREATE INDEX IF NOT EXISTS idx_companyfacts_raw_symbol ON sec_companyfacts_raw(symbol);
CREATE INDEX IF NOT EXISTS idx_companyfacts_raw_cik ON sec_companyfacts_raw(cik);
CREATE INDEX IF NOT EXISTS idx_companyfacts_raw_fetched ON sec_companyfacts_raw(fetched_at);

-- GIN index for JSONB queries (optional, for advanced queries)
CREATE INDEX IF NOT EXISTS idx_companyfacts_raw_facts_gin
    ON sec_companyfacts_raw USING GIN (companyfacts);

COMMENT ON TABLE sec_companyfacts_raw IS 'Stores raw SEC API responses with us-gaap structure (immutable)';
COMMENT ON COLUMN sec_companyfacts_raw.companyfacts IS 'Exact SEC API JSON response with facts.us-gaap structure';
COMMENT ON CONSTRAINT sec_companyfacts_raw_us_gaap_check ON sec_companyfacts_raw IS 'Ensures us-gaap structure exists (prevents flattened data)';

-- =============================================================================
-- TABLE 2: sec_companyfacts_processed (FAST QUERY CACHE - FLATTENED DATA)
-- =============================================================================

CREATE TABLE IF NOT EXISTS sec_companyfacts_processed (
    id                  SERIAL PRIMARY KEY,
    symbol              VARCHAR(10) NOT NULL,
    cik                 VARCHAR(10) NOT NULL,
    fiscal_year         INTEGER NOT NULL,
    fiscal_period       VARCHAR(5) NOT NULL,  -- Q1, Q2, Q3, Q4, FY

    -- FLATTENED FINANCIAL DATA (snake_case for Python compatibility)
    total_revenue       NUMERIC(20, 2),
    net_income          NUMERIC(20, 2),
    gross_profit        NUMERIC(20, 2),
    operating_income    NUMERIC(20, 2),
    cost_of_revenue     NUMERIC(20, 2),

    -- BALANCE SHEET
    total_assets        NUMERIC(20, 2),
    total_liabilities   NUMERIC(20, 2),
    current_assets      NUMERIC(20, 2),
    current_liabilities NUMERIC(20, 2),
    stockholders_equity NUMERIC(20, 2),
    accounts_receivable NUMERIC(20, 2),
    inventory           NUMERIC(20, 2),
    cash_and_equivalents NUMERIC(20, 2),

    -- DEBT
    long_term_debt      NUMERIC(20, 2),
    short_term_debt     NUMERIC(20, 2),
    total_debt          NUMERIC(20, 2),

    -- CASH FLOW
    operating_cash_flow NUMERIC(20, 2),
    capital_expenditures NUMERIC(20, 2),
    free_cash_flow      NUMERIC(20, 2),

    -- PRE-CALCULATED RATIOS (avoid recalculation)
    current_ratio       NUMERIC(10, 4),
    quick_ratio         NUMERIC(10, 4),
    debt_to_equity      NUMERIC(10, 4),
    roa                 NUMERIC(10, 4),
    roe                 NUMERIC(10, 4),
    gross_margin        NUMERIC(10, 4),
    operating_margin    NUMERIC(10, 4),
    net_margin          NUMERIC(10, 4),

    -- FILING METADATA
    adsh                VARCHAR(20) NOT NULL,  -- Accession number (unique per filing)
    form_type           VARCHAR(10),           -- 10-K, 10-Q, 20-F
    filed_date          DATE,
    period_end_date     DATE,
    frame               VARCHAR(20),           -- CY2024Q3, etc.

    -- PROCESSING METADATA
    extracted_at        TIMESTAMP NOT NULL DEFAULT NOW(),
    extraction_version  VARCHAR(50),
    data_quality_score  NUMERIC(5, 2),        -- 0-100 completeness score

    -- LINEAGE (links back to raw data for re-processing)
    raw_data_id         INTEGER REFERENCES sec_companyfacts_raw(id) ON DELETE CASCADE,

    -- CONSTRAINTS
    CONSTRAINT sec_companyfacts_processed_unique
        UNIQUE (symbol, fiscal_year, fiscal_period, adsh)
);

-- INDEXES for fast queries
CREATE INDEX IF NOT EXISTS idx_companyfacts_processed_symbol ON sec_companyfacts_processed(symbol);
CREATE INDEX IF NOT EXISTS idx_companyfacts_processed_period ON sec_companyfacts_processed(fiscal_year, fiscal_period);
CREATE INDEX IF NOT EXISTS idx_companyfacts_processed_adsh ON sec_companyfacts_processed(adsh);
CREATE INDEX IF NOT EXISTS idx_companyfacts_processed_extracted ON sec_companyfacts_processed(extracted_at);
CREATE INDEX IF NOT EXISTS idx_companyfacts_processed_symbol_year ON sec_companyfacts_processed(symbol, fiscal_year DESC);

COMMENT ON TABLE sec_companyfacts_processed IS 'Pre-processed quarterly/annual data (one row per filing)';
COMMENT ON COLUMN sec_companyfacts_processed.raw_data_id IS 'Links to raw data for re-processing without API re-fetch';

-- =============================================================================
-- TABLE 3: sec_companyfacts_metadata (CACHE CONTROL & DATA QUALITY TRACKING)
-- =============================================================================

CREATE TABLE IF NOT EXISTS sec_companyfacts_metadata (
    symbol              VARCHAR(10) PRIMARY KEY,
    cik                 VARCHAR(10) NOT NULL,
    entity_name         VARCHAR(255),

    -- CACHE CONTROL
    last_fetched        TIMESTAMP,
    last_processed      TIMESTAMP,
    fetch_count         INTEGER DEFAULT 0,
    cache_ttl_days      INTEGER DEFAULT 90,
    next_refresh_due    TIMESTAMP,

    -- DATA QUALITY
    raw_data_complete   BOOLEAN DEFAULT FALSE,
    processing_status   VARCHAR(20) DEFAULT 'pending',  -- pending, processing, completed, failed
    processing_error    TEXT,
    data_quality_grade  VARCHAR(2),                     -- A, B, C, D, F
    core_metrics_count  INTEGER DEFAULT 0,
    ratio_metrics_count INTEGER DEFAULT 0,

    -- FILING COVERAGE
    earliest_filing     DATE,
    latest_filing       DATE,
    total_filings       INTEGER DEFAULT 0,
    quarters_available  INTEGER DEFAULT 0,

    -- SYSTEM METADATA
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW()
);

-- INDEXES
CREATE INDEX IF NOT EXISTS idx_companyfacts_metadata_symbol ON sec_companyfacts_metadata(symbol);
CREATE INDEX IF NOT EXISTS idx_companyfacts_metadata_refresh ON sec_companyfacts_metadata(next_refresh_due);
CREATE INDEX IF NOT EXISTS idx_companyfacts_metadata_status ON sec_companyfacts_metadata(processing_status);

COMMENT ON TABLE sec_companyfacts_metadata IS 'Tracks cache freshness, data quality, and processing status';

-- =============================================================================
-- MIGRATION: CLEAR OLD DATA (old sec_companyfacts table has mixed data)
-- =============================================================================

-- Option 1: Backup old table (recommended)
-- CREATE TABLE sec_companyfacts_backup_20251102 AS SELECT * FROM sec_companyfacts;

-- Option 2: Clear all data (safest for fixing corruption)
-- WARNING: This will force all tickers to re-fetch from SEC API on next analysis
TRUNCATE TABLE sec_companyfacts CASCADE;

-- =============================================================================
-- VERIFICATION QUERIES
-- =============================================================================

-- Verify tables created
SELECT table_name, table_type
FROM information_schema.tables
WHERE table_name LIKE 'sec_companyfacts%'
ORDER BY table_name;

-- Check constraints
SELECT conname, contype, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = 'sec_companyfacts_raw'::regclass;

-- Sample query for future use (after data populated)
-- SELECT symbol, fiscal_year, fiscal_period, total_revenue, current_ratio
-- FROM sec_companyfacts_processed
-- WHERE symbol = 'AAPL'
-- ORDER BY fiscal_year DESC, fiscal_period DESC
-- LIMIT 8;
