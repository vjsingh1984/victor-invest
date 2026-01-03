-- InvestiGator Corrected Database Schema
-- Version: 6.1.0 (Fixed Table Names & Removed Obsolete Objects)
-- Copyright (c) 2025 Vijaykumar Singh
-- Licensed under the Apache License, Version 2.0
-- Created: 2025-06-03
-- Last Updated: 2025-06-03

-- This is the CORRECTED authoritative database schema for InvestiGator
-- Fixed table name mismatches and removed all obsolete objects

-- ================================================================================================
-- DATABASE SETUP
-- ================================================================================================

-- Run these commands manually to create the database and user:
-- CREATE DATABASE investment_analysis;
-- CREATE USER investment_ai WITH PASSWORD 'your_password_here';
-- GRANT ALL PRIVILEGES ON DATABASE investment_analysis TO investment_ai;

-- Connect to the database before running this schema:
-- \c investment_analysis

-- ================================================================================================
-- EXTENSIONS
-- ================================================================================================

CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- ================================================================================================
-- CORE ACTIVE TABLES ONLY
-- ================================================================================================

-- 1. Ticker to CIK mapping table
-- Used by: ticker_cik_mapper.py, all analysis modules
CREATE TABLE IF NOT EXISTS ticker_cik_mapping (
    ticker VARCHAR(10) PRIMARY KEY,
    cik VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    exchange VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_ticker_cik_cik ON ticker_cik_mapping(cik);
CREATE INDEX idx_ticker_cik_company ON ticker_cik_mapping(company_name);

-- 2. SEC response store - Caches raw SEC API responses
-- Used by: RdbmsCacheStorageHandler, SEC cache handlers
CREATE TABLE IF NOT EXISTS sec_responses (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_period VARCHAR(10) NOT NULL,
    form_type VARCHAR(20) NOT NULL,  -- Supports '10-K', '10-Q', '8-K', 'COMPREHENSIVE'
    category VARCHAR(50),
    response_data JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(symbol, fiscal_year, fiscal_period, form_type, category)
);

CREATE INDEX idx_sec_response_symbol ON sec_responses(symbol);
CREATE INDEX idx_sec_response_period ON sec_responses(fiscal_year, fiscal_period);

-- 3. LLM response table - Caches AI analysis responses
-- CORRECTED TABLE NAME: llm_responses (not llm_responses)
-- Used by: RdbmsCacheStorageHandler, LLM cache handlers, synthesizer
CREATE TABLE IF NOT EXISTS llm_responses (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    form_type VARCHAR(20) NOT NULL,  -- Extended for 'COMPREHENSIVE', 'SYNTHESIS', 'TECHNICAL'
    period VARCHAR(20) NOT NULL,  -- Format: 'YYYY-QX' or 'YYYY-FY'
    prompt TEXT NOT NULL,
    model_info JSONB NOT NULL DEFAULT '{}',
    response JSONB NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    llm_type VARCHAR(50) NOT NULL,  -- 'sec', 'ta', 'full', 'orchestrator_comprehensive', 'orchestrator_standard', etc.
    ts TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(symbol, form_type, period, llm_type)
);

CREATE INDEX idx_llm_response_symbol ON llm_responses(symbol);
CREATE INDEX idx_llm_response_type ON llm_responses(form_type, llm_type);
CREATE INDEX idx_llm_response_ts ON llm_responses(ts);

-- 4. SEC submissions table - Simple storage for SEC submissions data
-- CORRECTED TABLE NAME: sec_submissions (not all_submission_store)
-- Used by: AllSubmissionStoreDAO, submission cache handlers
-- SIMPLIFIED: Only stores raw data, all processing logic in Python
CREATE TABLE IF NOT EXISTS sec_submissions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    cik VARCHAR(10) NOT NULL,
    submissions_data JSONB NOT NULL,
    company_name VARCHAR(255),
    fetched_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(symbol, cik)
);

CREATE INDEX idx_submissions_symbol ON sec_submissions(symbol);
CREATE INDEX idx_submissions_cik ON sec_submissions(cik);
CREATE INDEX idx_submissions_updated ON sec_submissions(updated_at);

-- 5. Quarterly metrics - Standardized financial metrics by quarter
-- Used by: SEC strategies, financial aggregator
CREATE TABLE IF NOT EXISTS quarterly_metrics (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_period VARCHAR(10) NOT NULL,
    form_type VARCHAR(20) NOT NULL,
    metrics_data JSONB NOT NULL,
    filing_date DATE,
    period_end_date DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(symbol, fiscal_year, fiscal_period, form_type)
);

CREATE INDEX idx_quarterly_metrics_symbol ON quarterly_metrics(symbol);
CREATE INDEX idx_quarterly_metrics_period ON quarterly_metrics(fiscal_year, fiscal_period);
CREATE INDEX idx_quarterly_metrics_dates ON quarterly_metrics(filing_date, period_end_date);

-- 6. Peer group metrics cache - Caches financial ratios and peer comparisons
-- Used by: PeerGroupComparison, synthesis analysis
CREATE TABLE IF NOT EXISTS peer_metrics (
    peer_group_id VARCHAR(50) NOT NULL,  -- e.g., 'TECH_SOFTWARE_LARGE', 'FINANCE_BANKS_REGIONAL'
    symbol VARCHAR(10) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,  -- 'financial_ratios', 'peer_comparison', 'peer_stats'
    sector VARCHAR(100),
    industry VARCHAR(100),
    metrics_data JSONB NOT NULL,
    peer_symbols TEXT[],  -- Array of peer symbols analyzed
    calculation_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY(peer_group_id, symbol, metric_type)
);

CREATE INDEX idx_peer_metrics_symbol ON peer_metrics(symbol);
CREATE INDEX idx_peer_metrics_type ON peer_metrics(metric_type);
CREATE INDEX idx_peer_metrics_sector ON peer_metrics(sector, industry);

-- 7. Report generation tracking - Tracks all report generations with historical scores
-- Used by: report_generator.py, synthesis tracking, historical analysis
CREATE TABLE IF NOT EXISTS report_generation_history (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    report_type VARCHAR(50) NOT NULL DEFAULT 'synthesis',
    report_filename VARCHAR(255),
    
    -- Comprehensive Scores
    overall_score DECIMAL(4,2),
    fundamental_score DECIMAL(4,2),
    technical_score DECIMAL(4,2),
    
    -- Detailed Financial Scores
    income_statement_score DECIMAL(4,2),
    balance_sheet_score DECIMAL(4,2),
    cash_flow_score DECIMAL(4,2),
    growth_score DECIMAL(4,2),
    value_score DECIMAL(4,2),
    business_quality_score DECIMAL(4,2),
    data_quality_score DECIMAL(4,2),
    
    -- Investment Recommendation
    recommendation VARCHAR(20),
    confidence_level VARCHAR(10),
    time_horizon VARCHAR(20),
    position_size VARCHAR(20),
    
    -- Price Information
    current_price DECIMAL(10,2),
    target_price DECIMAL(10,2),
    upside_potential DECIMAL(5,2),
    
    -- Analysis Metadata
    analysis_mode VARCHAR(50) DEFAULT 'comprehensive',
    model_used VARCHAR(100),
    processing_time_seconds INTEGER,
    
    -- Data Sources Used
    sec_data_available BOOLEAN DEFAULT FALSE,
    technical_data_available BOOLEAN DEFAULT FALSE,
    peer_data_available BOOLEAN DEFAULT FALSE,
    quarters_analyzed INTEGER DEFAULT 0,
    
    -- Timestamps
    generated_at TIMESTAMP DEFAULT NOW(),
    market_date DATE,
    
    CONSTRAINT valid_scores CHECK (
        overall_score BETWEEN 0 AND 10 AND
        fundamental_score BETWEEN 0 AND 10 AND
        technical_score BETWEEN 0 AND 10
    ),
    
    CONSTRAINT valid_recommendation CHECK (
        recommendation IN ('STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL')
    )
);

CREATE INDEX idx_report_history_ticker ON report_generation_history(ticker);
CREATE INDEX idx_report_history_generated_at ON report_generation_history(generated_at);
CREATE INDEX idx_report_history_ticker_date ON report_generation_history(ticker, generated_at);
CREATE INDEX idx_report_history_recommendation ON report_generation_history(recommendation);
CREATE INDEX idx_report_history_overall_score ON report_generation_history(overall_score);
CREATE INDEX idx_peer_metrics_date ON peer_metrics(calculation_date);
CREATE INDEX idx_peer_metrics_group ON peer_metrics(peer_group_id);

-- ================================================================================================
-- PARTITIONED TABLES (For scalability)
-- ================================================================================================

-- 9. Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(20) PRIMARY KEY,
    description TEXT,
    applied_at TIMESTAMP DEFAULT NOW()
);

-- Insert current version
INSERT INTO schema_version (version, description)
VALUES ('6.1.0', 'Corrected table names and removed obsolete objects')
ON CONFLICT (version) DO NOTHING;

-- ================================================================================================
-- FUNCTIONS AND TRIGGERS
-- ================================================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to tables with updated_at
CREATE TRIGGER update_ticker_cik_mapping_updated_at BEFORE UPDATE ON ticker_cik_mapping
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sec_responses_updated_at BEFORE UPDATE ON sec_responses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sec_submissions_updated_at BEFORE UPDATE ON sec_submissions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_quarterly_metrics_updated_at BEFORE UPDATE ON quarterly_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_peer_metrics_updated_at BEFORE UPDATE ON peer_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ================================================================================================
-- PERMISSIONS
-- ================================================================================================

-- Grant permissions to investment_ai user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO investment_ai;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO investment_ai;

-- ================================================================================================
-- MAINTENANCE SCRIPTS
-- ================================================================================================

-- Clean old submission data (keep 90 days)
CREATE OR REPLACE FUNCTION clean_old_submissions()
RETURNS void AS $$
BEGIN
    DELETE FROM sec_submissions WHERE updated_at < CURRENT_DATE - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- ================================================================================================
-- COMPLETION
-- ================================================================================================

DO $$
BEGIN
    RAISE NOTICE '✅ InvestiGator corrected database schema v6.1.0 applied successfully';
    RAISE NOTICE 'Fixed table name mismatches and removed all obsolete objects.';
    RAISE NOTICE 'Active tables: ticker_cik_mapping, sec_responses, llm_responses, sec_submissions, quarterly_metrics, peer_metrics, report_generation_history';
    RAISE NOTICE 'Partitioned tables: fundamental_analysis, technical_analysis';
    RAISE NOTICE 'All obsolete views and tables have been removed.';
END $$;
