-- InvestiGator Sentiment Data Tables
-- Version: 7.0.0
-- RDBMS-Agnostic: Works with PostgreSQL and SQLite
-- Copyright (c) 2025 Vijaykumar Singh
-- Licensed under Apache License 2.0
--
-- Tables for: short_interest, insider_transactions, institutional_holdings

-- ============================================================================
-- SHORT INTEREST
-- ============================================================================

CREATE TABLE IF NOT EXISTS short_interest (
    symbol TEXT NOT NULL,
    settlement_date TEXT NOT NULL,
    short_interest INTEGER,
    avg_daily_volume INTEGER,
    days_to_cover REAL,
    short_interest_ratio REAL,
    short_percent_float REAL,  -- Short interest as % of float
    shares_outstanding INTEGER,
    squeeze_potential INTEGER DEFAULT 0,
    short_interest_change_pct REAL,
    prev_settlement_date TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, settlement_date)
);

CREATE INDEX IF NOT EXISTS idx_short_interest_symbol ON short_interest(symbol);
CREATE INDEX IF NOT EXISTS idx_short_interest_date ON short_interest(settlement_date DESC);

-- ============================================================================
-- FORM 4 FILINGS (Insider Transactions)
-- ============================================================================

CREATE TABLE IF NOT EXISTS form4_filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    filing_date TEXT,
    transaction_date TEXT,
    -- Owner information (aligned with data source code expectations)
    owner_name TEXT,  -- Used by insider_transactions data source
    owner_title TEXT,  -- Used by insider_transactions data source
    reporting_owner_name TEXT,  -- Alternative column name (legacy code)
    reporting_owner_title TEXT,  -- Alternative column name (legacy code)
    insider_name TEXT,  -- Alternative column name
    insider_title TEXT,  -- Alternative column name
    -- Role flags
    is_director INTEGER DEFAULT 0,
    is_officer INTEGER DEFAULT 0,
    is_ten_percent_owner INTEGER DEFAULT 0,
    -- Transaction details
    transaction_code TEXT,  -- P=Purchase, S=Sale, A=Award, etc.
    shares REAL,
    price_per_share REAL,
    total_value REAL,
    shares_owned_after REAL,
    transaction_data TEXT,  -- JSON blob for detailed transaction data
    is_significant INTEGER DEFAULT 0,  -- Flagged as significant transaction
    -- Filing metadata
    accession_number TEXT,
    form_type TEXT DEFAULT '4',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE (accession_number, owner_name, transaction_date)
);

CREATE INDEX IF NOT EXISTS idx_form4_symbol ON form4_filings(symbol);
CREATE INDEX IF NOT EXISTS idx_form4_date ON form4_filings(transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_form4_filing_date ON form4_filings(filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_form4_owner ON form4_filings(owner_name);

-- ============================================================================
-- INSIDER SENTIMENT (Aggregated)
-- ============================================================================

CREATE TABLE IF NOT EXISTS insider_sentiment (
    symbol TEXT NOT NULL,
    calculation_date TEXT NOT NULL,
    period_days INTEGER NOT NULL,
    buy_count INTEGER DEFAULT 0,
    sell_count INTEGER DEFAULT 0,
    buy_value REAL DEFAULT 0,
    sell_value REAL DEFAULT 0,
    sentiment_score REAL,
    cluster_detected INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, calculation_date, period_days)
);

CREATE INDEX IF NOT EXISTS idx_insider_sentiment_symbol ON insider_sentiment(symbol);
CREATE INDEX IF NOT EXISTS idx_insider_sentiment_date ON insider_sentiment(calculation_date DESC);

-- ============================================================================
-- INSTITUTIONS (13F Filers)
-- ============================================================================

CREATE TABLE IF NOT EXISTS institutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cik TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_institutions_cik ON institutions(cik);
CREATE INDEX IF NOT EXISTS idx_institutions_name ON institutions(name);

-- ============================================================================
-- FORM 13F FILINGS
-- ============================================================================

CREATE TABLE IF NOT EXISTS form13f_filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER REFERENCES institutions(id),
    accession_number TEXT UNIQUE NOT NULL,
    report_quarter TEXT,  -- Date as TEXT for SQLite compatibility
    filing_date TEXT,
    total_value REAL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_13f_filings_institution ON form13f_filings(institution_id);
CREATE INDEX IF NOT EXISTS idx_13f_filings_quarter ON form13f_filings(report_quarter DESC);

-- ============================================================================
-- FORM 13F HOLDINGS
-- ============================================================================

CREATE TABLE IF NOT EXISTS form13f_holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id INTEGER REFERENCES form13f_filings(id),
    symbol TEXT,
    cusip TEXT,
    issuer_name TEXT,  -- Issuer company name
    class_title TEXT,  -- Security class (e.g., "COM", "CL A")
    shares INTEGER,
    value_thousands INTEGER,
    put_call TEXT,
    investment_discretion TEXT,
    -- Denormalized columns for efficient queries
    filer_cik TEXT,  -- CIK of the filing institution
    report_date TEXT,  -- Quarter end date
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE (filing_id, cusip)
);

CREATE INDEX IF NOT EXISTS idx_13f_holdings_symbol ON form13f_holdings(symbol);
CREATE INDEX IF NOT EXISTS idx_13f_holdings_filing ON form13f_holdings(filing_id);
CREATE INDEX IF NOT EXISTS idx_13f_holdings_filer_cik ON form13f_holdings(filer_cik);
CREATE INDEX IF NOT EXISTS idx_13f_holdings_report_date ON form13f_holdings(report_date DESC);

-- ============================================================================
-- FORM 13F FILERS (Alternative to institutions, used by legacy code)
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
-- INSTITUTIONAL OWNERSHIP (Aggregated)
-- ============================================================================

CREATE TABLE IF NOT EXISTS institutional_ownership (
    symbol TEXT NOT NULL,
    report_quarter TEXT NOT NULL,
    num_institutions INTEGER,
    total_institutional_shares INTEGER,
    total_institutional_value REAL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, report_quarter)
);

CREATE INDEX IF NOT EXISTS idx_inst_ownership_symbol ON institutional_ownership(symbol);

-- Insert version
INSERT OR IGNORE INTO schema_version (version, description)
VALUES ('7.0.3', 'Sentiment tables - short_interest, form4, form13f, form13f_filers, insider_sentiment');
