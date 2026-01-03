-- InvestiGator Market Data Tables
-- Version: 7.0.0
-- RDBMS-Agnostic: Works with PostgreSQL and SQLite
-- Copyright (c) 2025 Vijaykumar Singh
-- Licensed under Apache License 2.0
--
-- Tables for: price_history, technical_indicators, symbol metadata

-- ============================================================================
-- SYMBOL METADATA (Master symbol table)
-- ============================================================================

CREATE TABLE IF NOT EXISTS symbol (
    ticker TEXT PRIMARY KEY,
    company_name TEXT,
    exchange TEXT,
    sector TEXT,
    industry TEXT,
    market_cap REAL,
    shares_outstanding REAL,
    beta REAL,
    pe_ratio REAL,
    forward_pe REAL,
    dividend_yield REAL,
    eps_ttm REAL,
    revenue_ttm REAL,
    profit_margin REAL,
    operating_margin REAL,
    roe REAL,
    roa REAL,
    debt_to_equity REAL,
    current_ratio REAL,
    quick_ratio REAL,
    book_value_per_share REAL,
    price_to_book REAL,
    price_to_sales REAL,
    ev_to_ebitda REAL,
    free_cash_flow REAL,
    revenue_growth REAL,
    earnings_growth REAL,
    analyst_target_price REAL,
    analyst_recommendation TEXT,
    is_active INTEGER DEFAULT 1,
    is_sp500 INTEGER DEFAULT 0,
    is_russell1000 INTEGER DEFAULT 0,
    sic_code TEXT,
    cik TEXT,
    cusip TEXT,
    last_quote_date TEXT,
    last_quote_price REAL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_symbol_sector ON symbol(sector);
CREATE INDEX IF NOT EXISTS idx_symbol_industry ON symbol(industry);
CREATE INDEX IF NOT EXISTS idx_symbol_market_cap ON symbol(market_cap);
CREATE INDEX IF NOT EXISTS idx_symbol_active ON symbol(is_active);
CREATE INDEX IF NOT EXISTS idx_symbol_sp500 ON symbol(is_sp500);
CREATE INDEX IF NOT EXISTS idx_symbol_russell1000 ON symbol(is_russell1000);

-- ============================================================================
-- TICKER DATA (Historical OHLCV prices)
-- ============================================================================

CREATE TABLE IF NOT EXISTS tickerdata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL NOT NULL,
    volume INTEGER,
    adj_close REAL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_tickerdata_ticker ON tickerdata(ticker);
CREATE INDEX IF NOT EXISTS idx_tickerdata_date ON tickerdata(date DESC);
CREATE INDEX IF NOT EXISTS idx_tickerdata_ticker_date ON tickerdata(ticker, date DESC);

-- ============================================================================
-- TECHNICAL INDICATORS (Cached calculations)
-- ============================================================================

CREATE TABLE IF NOT EXISTS technical_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    calculation_date TEXT NOT NULL,
    -- Trend Indicators
    sma_20 REAL,
    sma_50 REAL,
    sma_200 REAL,
    ema_12 REAL,
    ema_26 REAL,
    -- Momentum Indicators
    rsi_14 REAL,
    macd REAL,
    macd_signal REAL,
    macd_histogram REAL,
    stoch_k REAL,
    stoch_d REAL,
    williams_r REAL,
    -- Volatility Indicators
    atr_14 REAL,
    bollinger_upper REAL,
    bollinger_middle REAL,
    bollinger_lower REAL,
    bollinger_width REAL,
    -- Volume Indicators
    obv REAL,
    mfi_14 REAL,
    adx_14 REAL,
    plus_di REAL,
    minus_di REAL,
    -- Price Context
    current_price REAL,
    price_vs_sma20 REAL,
    price_vs_sma50 REAL,
    price_vs_sma200 REAL,
    -- Signals
    trend_signal TEXT,  -- bullish, bearish, neutral
    momentum_signal TEXT,
    volatility_regime TEXT,  -- low, normal, high
    -- Metadata
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, calculation_date)
);

CREATE INDEX IF NOT EXISTS idx_tech_indicators_symbol ON technical_indicators(symbol);
CREATE INDEX IF NOT EXISTS idx_tech_indicators_date ON technical_indicators(calculation_date DESC);

-- ============================================================================
-- SHARES HISTORY (For split detection and normalization)
-- ============================================================================

CREATE TABLE IF NOT EXISTS shares_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    record_date TEXT NOT NULL,
    shares_outstanding REAL,
    split_factor REAL DEFAULT 1.0,
    split_date TEXT,
    split_ratio TEXT,  -- e.g., "4:1", "1:10"
    source TEXT,  -- sec, yahoo, manual
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, record_date)
);

CREATE INDEX IF NOT EXISTS idx_shares_history_symbol ON shares_history(symbol);
CREATE INDEX IF NOT EXISTS idx_shares_history_date ON shares_history(record_date DESC);

-- Insert version
INSERT OR IGNORE INTO schema_version (version, description)
VALUES ('7.0.1', 'Market data tables - tickerdata, symbol, technical_indicators');
