-- Migration 006: Add missing columns for data sources
-- Created: 2025-01-03
-- Description: Adds columns required by data source implementations
--
-- This migration adds missing columns that the data source code expects:
-- 1. form4_filings: owner_name, owner_title, is_director, is_officer
-- 2. Ensures all tables exist with correct structure

-- ============================================================================
-- FORM4_FILINGS - Add missing columns for insider_transactions data source
-- ============================================================================

-- Add owner_name if it doesn't exist (alias for insider_name)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'form4_filings' AND column_name = 'owner_name'
    ) THEN
        ALTER TABLE form4_filings ADD COLUMN owner_name VARCHAR(200);
        -- Copy data from insider_name if it exists
        UPDATE form4_filings SET owner_name = insider_name WHERE owner_name IS NULL AND insider_name IS NOT NULL;
    END IF;
END $$;

-- Add owner_title if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'form4_filings' AND column_name = 'owner_title'
    ) THEN
        ALTER TABLE form4_filings ADD COLUMN owner_title VARCHAR(200);
        -- Copy data from insider_title if it exists
        UPDATE form4_filings SET owner_title = insider_title WHERE owner_title IS NULL AND insider_title IS NOT NULL;
    END IF;
END $$;

-- Add is_director if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'form4_filings' AND column_name = 'is_director'
    ) THEN
        ALTER TABLE form4_filings ADD COLUMN is_director BOOLEAN DEFAULT FALSE;
    END IF;
END $$;

-- Add is_officer if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'form4_filings' AND column_name = 'is_officer'
    ) THEN
        ALTER TABLE form4_filings ADD COLUMN is_officer BOOLEAN DEFAULT FALSE;
    END IF;
END $$;

-- Add transaction_code if it doesn't exist (replaces transaction_type)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'form4_filings' AND column_name = 'transaction_code'
    ) THEN
        ALTER TABLE form4_filings ADD COLUMN transaction_code VARCHAR(10);
        -- Copy from transaction_type if exists
        UPDATE form4_filings SET transaction_code = transaction_type
        WHERE transaction_code IS NULL AND transaction_type IS NOT NULL;
    END IF;
END $$;

-- ============================================================================
-- QUARTERLY_METRICS - Add calculated_at column
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'quarterly_metrics' AND column_name = 'calculated_at'
    ) THEN
        ALTER TABLE quarterly_metrics ADD COLUMN calculated_at TIMESTAMP DEFAULT NOW();
    END IF;
END $$;

-- ============================================================================
-- TICKERDATA TABLE (if not exists)
-- ============================================================================

CREATE TABLE IF NOT EXISTS tickerdata (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(12, 4),
    high DECIMAL(12, 4),
    low DECIMAL(12, 4),
    close DECIMAL(12, 4) NOT NULL,
    volume BIGINT,
    adj_close DECIMAL(12, 4),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_tickerdata_ticker ON tickerdata(ticker);
CREATE INDEX IF NOT EXISTS idx_tickerdata_date ON tickerdata(date DESC);
CREATE INDEX IF NOT EXISTS idx_tickerdata_ticker_date ON tickerdata(ticker, date DESC);

-- ============================================================================
-- SYMBOL TABLE (if not exists)
-- ============================================================================

CREATE TABLE IF NOT EXISTS symbol (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255),
    exchange VARCHAR(50),
    sector VARCHAR(100),
    industry VARCHAR(200),
    market_cap DECIMAL(20, 2),
    shares_outstanding DECIMAL(20, 0),
    beta DECIMAL(6, 4),
    pe_ratio DECIMAL(10, 4),
    forward_pe DECIMAL(10, 4),
    dividend_yield DECIMAL(6, 4),
    eps_ttm DECIMAL(10, 4),
    revenue_ttm DECIMAL(20, 2),
    profit_margin DECIMAL(6, 4),
    operating_margin DECIMAL(6, 4),
    roe DECIMAL(6, 4),
    roa DECIMAL(6, 4),
    debt_to_equity DECIMAL(10, 4),
    current_ratio DECIMAL(10, 4),
    quick_ratio DECIMAL(10, 4),
    book_value_per_share DECIMAL(12, 4),
    price_to_book DECIMAL(10, 4),
    price_to_sales DECIMAL(10, 4),
    ev_to_ebitda DECIMAL(10, 4),
    free_cash_flow DECIMAL(20, 2),
    revenue_growth DECIMAL(6, 4),
    earnings_growth DECIMAL(6, 4),
    analyst_target_price DECIMAL(10, 4),
    analyst_recommendation VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    is_sp500 BOOLEAN DEFAULT FALSE,
    is_russell1000 BOOLEAN DEFAULT FALSE,
    sic_code VARCHAR(10),
    cik VARCHAR(20),
    cusip VARCHAR(9),
    last_quote_date DATE,
    last_quote_price DECIMAL(12, 4),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_symbol_sector ON symbol(sector);
CREATE INDEX IF NOT EXISTS idx_symbol_industry ON symbol(industry);
CREATE INDEX IF NOT EXISTS idx_symbol_market_cap ON symbol(market_cap);
CREATE INDEX IF NOT EXISTS idx_symbol_active ON symbol(is_active);
CREATE INDEX IF NOT EXISTS idx_symbol_sp500 ON symbol(is_sp500);
CREATE INDEX IF NOT EXISTS idx_symbol_russell1000 ON symbol(is_russell1000);

-- ============================================================================
-- TECHNICAL_INDICATORS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    calculation_date DATE NOT NULL,
    sma_20 DECIMAL(12, 4),
    sma_50 DECIMAL(12, 4),
    sma_200 DECIMAL(12, 4),
    ema_12 DECIMAL(12, 4),
    ema_26 DECIMAL(12, 4),
    rsi_14 DECIMAL(6, 2),
    macd DECIMAL(12, 4),
    macd_signal DECIMAL(12, 4),
    macd_histogram DECIMAL(12, 4),
    stoch_k DECIMAL(6, 2),
    stoch_d DECIMAL(6, 2),
    williams_r DECIMAL(6, 2),
    atr_14 DECIMAL(12, 4),
    bollinger_upper DECIMAL(12, 4),
    bollinger_middle DECIMAL(12, 4),
    bollinger_lower DECIMAL(12, 4),
    bollinger_width DECIMAL(12, 4),
    obv DECIMAL(20, 0),
    mfi_14 DECIMAL(6, 2),
    adx_14 DECIMAL(6, 2),
    plus_di DECIMAL(6, 2),
    minus_di DECIMAL(6, 2),
    current_price DECIMAL(12, 4),
    price_vs_sma20 DECIMAL(8, 4),
    price_vs_sma50 DECIMAL(8, 4),
    price_vs_sma200 DECIMAL(8, 4),
    trend_signal VARCHAR(20),
    momentum_signal VARCHAR(20),
    volatility_regime VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, calculation_date)
);

CREATE INDEX IF NOT EXISTS idx_tech_indicators_symbol ON technical_indicators(symbol);
CREATE INDEX IF NOT EXISTS idx_tech_indicators_date ON technical_indicators(calculation_date DESC);

-- ============================================================================
-- SHARES_HISTORY TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS shares_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    record_date DATE NOT NULL,
    shares_outstanding DECIMAL(20, 0),
    split_factor DECIMAL(10, 6) DEFAULT 1.0,
    split_date DATE,
    split_ratio VARCHAR(20),
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, record_date)
);

CREATE INDEX IF NOT EXISTS idx_shares_history_symbol ON shares_history(symbol);
CREATE INDEX IF NOT EXISTS idx_shares_history_date ON shares_history(record_date DESC);

-- ============================================================================
-- VERSION UPDATE
-- ============================================================================

INSERT INTO schema_version (version, description)
VALUES ('7.0.5', 'Migration 006: Add missing columns for data sources')
ON CONFLICT (version) DO NOTHING;

-- Summary
DO $$
BEGIN
    RAISE NOTICE '✅ Migration 006 complete: Added missing columns for data sources';
END $$;
