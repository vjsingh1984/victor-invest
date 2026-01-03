-- Migration 007: Fix data source schemas
-- Adds missing columns expected by data source queries
-- Version: 7.0.7
-- Date: 2025-01-03

-- ============================================================================
-- FIX: short_interest table - add short_percent_float
-- ============================================================================

ALTER TABLE short_interest ADD COLUMN IF NOT EXISTS short_percent_float REAL;

-- ============================================================================
-- FIX: form13f_holdings table - add denormalized columns for queries
-- The code expects direct columns rather than joins in some places
-- ============================================================================

ALTER TABLE form13f_holdings ADD COLUMN IF NOT EXISTS filer_cik TEXT;
ALTER TABLE form13f_holdings ADD COLUMN IF NOT EXISTS report_date TEXT;
ALTER TABLE form13f_holdings ADD COLUMN IF NOT EXISTS issuer_name TEXT;
ALTER TABLE form13f_holdings ADD COLUMN IF NOT EXISTS class_title TEXT;

-- Create index for efficient queries
CREATE INDEX IF NOT EXISTS idx_13f_holdings_filer_cik ON form13f_holdings(filer_cik);
CREATE INDEX IF NOT EXISTS idx_13f_holdings_report_date ON form13f_holdings(report_date DESC);

-- ============================================================================
-- CREATE: form13f_filers table (alternative to institutions, used by some code)
-- ============================================================================

CREATE TABLE IF NOT EXISTS form13f_filers (
    cik TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    filing_date TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_13f_filers_name ON form13f_filers(name);

-- ============================================================================
-- FIX: form4_filings table - add alternative column names
-- Some code uses reporting_owner_name instead of owner_name
-- ============================================================================

ALTER TABLE form4_filings ADD COLUMN IF NOT EXISTS reporting_owner_name TEXT;
ALTER TABLE form4_filings ADD COLUMN IF NOT EXISTS reporting_owner_title TEXT;
ALTER TABLE form4_filings ADD COLUMN IF NOT EXISTS transaction_data TEXT;
ALTER TABLE form4_filings ADD COLUMN IF NOT EXISTS is_significant INTEGER DEFAULT 0;

-- Populate reporting_owner_name from owner_name if empty
UPDATE form4_filings
SET reporting_owner_name = owner_name
WHERE reporting_owner_name IS NULL AND owner_name IS NOT NULL;

UPDATE form4_filings
SET reporting_owner_title = owner_title
WHERE reporting_owner_title IS NULL AND owner_title IS NOT NULL;

-- ============================================================================
-- FIX: quarterly_metrics table - ensure revenue column exists
-- ============================================================================

-- Note: quarterly_metrics may need to be created or have the revenue column added
-- This is a fallback if the table exists but lacks the column

-- First ensure the table exists
CREATE TABLE IF NOT EXISTS quarterly_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_period TEXT NOT NULL,
    -- Income Statement
    revenue REAL,
    cost_of_revenue REAL,
    gross_profit REAL,
    operating_income REAL,
    net_income REAL,
    ebitda REAL,
    -- Balance Sheet
    total_assets REAL,
    total_liabilities REAL,
    stockholders_equity REAL,
    cash_and_equivalents REAL,
    total_debt REAL,
    -- Cash Flow
    operating_cash_flow REAL,
    capital_expenditures REAL,
    free_cash_flow REAL,
    -- Per Share
    eps_basic REAL,
    eps_diluted REAL,
    book_value_per_share REAL,
    -- Shares
    shares_outstanding REAL,
    -- Metadata
    filing_date TEXT,
    source TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, fiscal_year, fiscal_period)
);

CREATE INDEX IF NOT EXISTS idx_quarterly_metrics_symbol ON quarterly_metrics(symbol);
CREATE INDEX IF NOT EXISTS idx_quarterly_metrics_period ON quarterly_metrics(fiscal_year, fiscal_period);

-- Update schema version
INSERT OR REPLACE INTO schema_version (version, description, applied_at)
VALUES ('7.0.7', 'Fix data source schemas - add missing columns', datetime('now'));
