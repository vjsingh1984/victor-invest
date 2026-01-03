-- InvestiGator Core Tables Schema
-- Version: 7.0.0
-- RDBMS-Agnostic: Works with PostgreSQL and SQLite
-- Copyright (c) 2025 Vijaykumar Singh
-- Licensed under Apache License 2.0
--
-- Usage:
--   PostgreSQL: psql -h HOST -U USER -d DATABASE -f schema/install/00_core_tables.sql
--   SQLite:     sqlite3 investigator.db < schema/install/00_core_tables.sql

-- ============================================================================
-- TICKER/CIK MAPPING
-- ============================================================================

CREATE TABLE IF NOT EXISTS ticker_cik_mapping (
    ticker TEXT PRIMARY KEY,
    cik TEXT NOT NULL,
    company_name TEXT,
    exchange TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ticker_cik_cik ON ticker_cik_mapping(cik);
CREATE INDEX IF NOT EXISTS idx_ticker_cik_company ON ticker_cik_mapping(company_name);

-- ============================================================================
-- SEC RESPONSES CACHE
-- ============================================================================

CREATE TABLE IF NOT EXISTS sec_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_period TEXT NOT NULL,
    form_type TEXT NOT NULL,
    category TEXT,
    response_data TEXT NOT NULL,  -- JSON
    metadata TEXT,  -- JSON
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, fiscal_year, fiscal_period, form_type, category)
);

CREATE INDEX IF NOT EXISTS idx_sec_response_symbol ON sec_responses(symbol);
CREATE INDEX IF NOT EXISTS idx_sec_response_period ON sec_responses(fiscal_year, fiscal_period);

-- ============================================================================
-- LLM RESPONSES CACHE
-- ============================================================================

CREATE TABLE IF NOT EXISTS llm_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    form_type TEXT NOT NULL,
    period TEXT NOT NULL,
    prompt TEXT NOT NULL,
    model_info TEXT NOT NULL DEFAULT '{}',  -- JSON
    response TEXT NOT NULL DEFAULT '{}',  -- JSON
    metadata TEXT DEFAULT '{}',  -- JSON
    llm_type TEXT NOT NULL DEFAULT 'unknown',
    ts TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, form_type, period, llm_type)
);

CREATE INDEX IF NOT EXISTS idx_llm_response_symbol ON llm_responses(symbol);
CREATE INDEX IF NOT EXISTS idx_llm_response_type ON llm_responses(form_type, llm_type);
CREATE INDEX IF NOT EXISTS idx_llm_response_ts ON llm_responses(ts);

-- ============================================================================
-- SEC SUBMISSIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS sec_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    cik TEXT NOT NULL,
    submissions_data TEXT NOT NULL,  -- JSON
    company_name TEXT,
    fetched_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, cik)
);

CREATE INDEX IF NOT EXISTS idx_submissions_symbol ON sec_submissions(symbol);
CREATE INDEX IF NOT EXISTS idx_submissions_cik ON sec_submissions(cik);
CREATE INDEX IF NOT EXISTS idx_submissions_updated ON sec_submissions(updated_at);

-- ============================================================================
-- SEC COMPANY FACTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS sec_companyfacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    cik TEXT NOT NULL,
    companyfacts TEXT NOT NULL,  -- JSON
    company_name TEXT,
    metadata TEXT,  -- JSON
    fetched_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, cik)
);

CREATE INDEX IF NOT EXISTS idx_companyfacts_symbol ON sec_companyfacts(symbol);
CREATE INDEX IF NOT EXISTS idx_companyfacts_cik ON sec_companyfacts(cik);
CREATE INDEX IF NOT EXISTS idx_companyfacts_updated ON sec_companyfacts(updated_at);

-- ============================================================================
-- QUARTERLY METRICS
-- ============================================================================

CREATE TABLE IF NOT EXISTS quarterly_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_period TEXT NOT NULL,
    form_type TEXT NOT NULL,
    metrics_data TEXT NOT NULL,  -- JSON with revenue, net_income, etc.
    filing_date TEXT,
    period_end_date TEXT,
    calculated_at TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, fiscal_year, fiscal_period, form_type)
);

CREATE INDEX IF NOT EXISTS idx_quarterly_metrics_symbol ON quarterly_metrics(symbol);
CREATE INDEX IF NOT EXISTS idx_quarterly_metrics_period ON quarterly_metrics(fiscal_year, fiscal_period);
CREATE INDEX IF NOT EXISTS idx_quarterly_metrics_dates ON quarterly_metrics(filing_date, period_end_date);

-- ============================================================================
-- PEER METRICS
-- ============================================================================

CREATE TABLE IF NOT EXISTS peer_metrics (
    peer_group_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    sector TEXT,
    industry TEXT,
    metrics_data TEXT NOT NULL,  -- JSON
    peer_symbols TEXT,  -- JSON array
    calculation_date TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY(peer_group_id, symbol, metric_type)
);

CREATE INDEX IF NOT EXISTS idx_peer_metrics_symbol ON peer_metrics(symbol);
CREATE INDEX IF NOT EXISTS idx_peer_metrics_type ON peer_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_peer_metrics_sector ON peer_metrics(sector, industry);
CREATE INDEX IF NOT EXISTS idx_peer_metrics_date ON peer_metrics(calculation_date);

-- ============================================================================
-- REPORT GENERATION HISTORY
-- ============================================================================

CREATE TABLE IF NOT EXISTS report_generation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    report_type TEXT NOT NULL DEFAULT 'synthesis',
    report_filename TEXT,
    overall_score REAL,
    fundamental_score REAL,
    technical_score REAL,
    income_statement_score REAL,
    balance_sheet_score REAL,
    cash_flow_score REAL,
    growth_score REAL,
    value_score REAL,
    business_quality_score REAL,
    data_quality_score REAL,
    recommendation TEXT,
    confidence_level TEXT,
    time_horizon TEXT,
    position_size TEXT,
    current_price REAL,
    target_price REAL,
    upside_potential REAL,
    analysis_mode TEXT DEFAULT 'comprehensive',
    model_used TEXT,
    processing_time_seconds INTEGER,
    sec_data_available INTEGER DEFAULT 0,
    technical_data_available INTEGER DEFAULT 0,
    peer_data_available INTEGER DEFAULT 0,
    quarters_analyzed INTEGER DEFAULT 0,
    generated_at TEXT DEFAULT (datetime('now')),
    market_date TEXT
);

CREATE INDEX IF NOT EXISTS idx_report_history_ticker ON report_generation_history(ticker);
CREATE INDEX IF NOT EXISTS idx_report_history_generated_at ON report_generation_history(generated_at);
CREATE INDEX IF NOT EXISTS idx_report_history_recommendation ON report_generation_history(recommendation);

-- ============================================================================
-- SCHEMA VERSION TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY,
    description TEXT,
    applied_at TEXT DEFAULT (datetime('now'))
);

-- Insert version
INSERT OR IGNORE INTO schema_version (version, description)
VALUES ('7.0.0', 'Core tables - RDBMS agnostic install');
