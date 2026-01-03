-- Scheduled Data Collection Tables
-- Run this to create tables required by the scheduled collectors
--
-- Usage:
--   psql -h HOST -U USER -d DATABASE -f schema/scheduled_tables.sql

BEGIN;

-- Treasury Yields (collect_treasury_data.py)
CREATE TABLE IF NOT EXISTS treasury_yields (
    date DATE PRIMARY KEY,
    yield_1m DECIMAL(6, 4),
    yield_3m DECIMAL(6, 4),
    yield_6m DECIMAL(6, 4),
    yield_1y DECIMAL(6, 4),
    yield_2y DECIMAL(6, 4),
    yield_5y DECIMAL(6, 4),
    yield_10y DECIMAL(6, 4),
    yield_20y DECIMAL(6, 4),
    yield_30y DECIMAL(6, 4),
    spread_10y_2y DECIMAL(6, 4),
    spread_10y_3m DECIMAL(6, 4),
    is_inverted BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_treasury_yields_date ON treasury_yields(date DESC);

-- Macro Indicators (refresh_macro_indicators.py)
CREATE TABLE IF NOT EXISTS macro_indicators (
    id SERIAL PRIMARY KEY,
    series_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200),
    category VARCHAR(50),
    frequency VARCHAR(20),
    units VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS macro_indicator_values (
    indicator_id INTEGER REFERENCES macro_indicators(id),
    date DATE NOT NULL,
    value DECIMAL(20, 6),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (indicator_id, date)
);
CREATE INDEX IF NOT EXISTS idx_macro_values_date ON macro_indicator_values(date DESC);

-- Insider Transactions (collect_insider_transactions.py)
CREATE TABLE IF NOT EXISTS form4_filings (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    filing_date DATE,
    transaction_date DATE,
    insider_name VARCHAR(200),
    insider_title VARCHAR(200),
    transaction_type VARCHAR(10),
    shares NUMERIC(20, 0),
    price_per_share DECIMAL(12, 4),
    total_value DECIMAL(20, 2),
    shares_owned_after NUMERIC(20, 0),
    accession_number VARCHAR(30),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (accession_number, insider_name, transaction_date)
);
CREATE INDEX IF NOT EXISTS idx_form4_symbol ON form4_filings(symbol);
CREATE INDEX IF NOT EXISTS idx_form4_date ON form4_filings(transaction_date DESC);

CREATE TABLE IF NOT EXISTS insider_sentiment (
    symbol VARCHAR(10) NOT NULL,
    calculation_date DATE NOT NULL,
    period_days INTEGER NOT NULL,
    buy_count INTEGER DEFAULT 0,
    sell_count INTEGER DEFAULT 0,
    buy_value DECIMAL(20, 2) DEFAULT 0,
    sell_value DECIMAL(20, 2) DEFAULT 0,
    sentiment_score DECIMAL(6, 4),
    cluster_detected BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (symbol, calculation_date, period_days)
);

-- Institutional Holdings (collect_13f_filings.py)
CREATE TABLE IF NOT EXISTS institutions (
    id SERIAL PRIMARY KEY,
    cik VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(300) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS form13f_filings (
    id SERIAL PRIMARY KEY,
    institution_id INTEGER REFERENCES institutions(id),
    accession_number VARCHAR(30) UNIQUE NOT NULL,
    report_quarter DATE,
    filing_date DATE,
    total_value DECIMAL(20, 2),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS form13f_holdings (
    id SERIAL PRIMARY KEY,
    filing_id INTEGER REFERENCES form13f_filings(id),
    symbol VARCHAR(10),
    cusip VARCHAR(9),
    shares NUMERIC(20, 0),
    value_thousands NUMERIC(20, 0),
    put_call VARCHAR(10),
    investment_discretion VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (filing_id, cusip)
);
CREATE INDEX IF NOT EXISTS idx_13f_holdings_symbol ON form13f_holdings(symbol);

CREATE TABLE IF NOT EXISTS institutional_ownership (
    symbol VARCHAR(10) NOT NULL,
    report_quarter DATE NOT NULL,
    num_institutions INTEGER,
    total_institutional_shares NUMERIC(20, 0),
    total_institutional_value DECIMAL(20, 2),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (symbol, report_quarter)
);

-- Short Interest (collect_short_interest.py)
CREATE TABLE IF NOT EXISTS short_interest (
    symbol VARCHAR(10) NOT NULL,
    settlement_date DATE NOT NULL,
    short_interest NUMERIC(20, 0),
    avg_daily_volume NUMERIC(20, 0),
    days_to_cover DECIMAL(10, 2),
    short_interest_ratio DECIMAL(10, 4),
    shares_outstanding NUMERIC(20, 0),
    squeeze_potential BOOLEAN DEFAULT FALSE,
    short_interest_change_pct DECIMAL(10, 4),
    prev_settlement_date DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (symbol, settlement_date)
);
CREATE INDEX IF NOT EXISTS idx_short_interest_symbol ON short_interest(symbol);

-- Market Regime (update_market_regime.py)
CREATE TABLE IF NOT EXISTS market_regime_history (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE UNIQUE NOT NULL,
    regime VARCHAR(50),
    credit_cycle_phase VARCHAR(50),
    volatility_regime VARCHAR(50),
    recession_probability DECIMAL(6, 4),
    yield_curve_inverted BOOLEAN DEFAULT FALSE,
    vix_level DECIMAL(8, 2),
    credit_spread DECIMAL(8, 4),
    risk_off_signal BOOLEAN DEFAULT FALSE,
    recommendations TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS current_market_regime (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    regime VARCHAR(50),
    credit_cycle_phase VARCHAR(50),
    volatility_regime VARCHAR(50),
    recession_probability DECIMAL(6, 4),
    yield_curve_inverted BOOLEAN DEFAULT FALSE,
    risk_off_signal BOOLEAN DEFAULT FALSE,
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS regime_transitions (
    id SERIAL PRIMARY KEY,
    transition_date DATE NOT NULL,
    transition_type VARCHAR(50) NOT NULL,
    from_state VARCHAR(50),
    to_state VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (transition_date, transition_type)
);

-- Credit Risk Scores (calculate_credit_risk.py)
CREATE TABLE IF NOT EXISTS credit_risk_scores (
    symbol VARCHAR(10) NOT NULL,
    calculation_date DATE NOT NULL,
    altman_z_score DECIMAL(10, 4),
    altman_z_interpretation VARCHAR(50),
    beneish_m_score DECIMAL(10, 4),
    beneish_m_interpretation VARCHAR(50),
    piotroski_f_score INTEGER,
    piotroski_f_interpretation VARCHAR(50),
    distress_tier VARCHAR(30),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (symbol, calculation_date)
);
CREATE INDEX IF NOT EXISTS idx_credit_risk_symbol ON credit_risk_scores(symbol);

CREATE TABLE IF NOT EXISTS current_credit_risk (
    symbol VARCHAR(10) PRIMARY KEY,
    altman_z_score DECIMAL(10, 4),
    beneish_m_score DECIMAL(10, 4),
    piotroski_f_score INTEGER,
    distress_tier VARCHAR(30),
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS credit_risk_summary (
    summary_date DATE PRIMARY KEY,
    total_analyzed INTEGER,
    healthy_count INTEGER DEFAULT 0,
    watch_count INTEGER DEFAULT 0,
    concern_count INTEGER DEFAULT 0,
    distressed_count INTEGER DEFAULT 0,
    severe_distress_count INTEGER DEFAULT 0,
    avg_z_score DECIMAL(10, 4),
    avg_m_score DECIMAL(10, 4),
    avg_f_score DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Scheduler Job Runs (for monitoring)
CREATE TABLE IF NOT EXISTS scheduler_job_runs (
    id SERIAL PRIMARY KEY,
    job_name VARCHAR(100) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_seconds DECIMAL(10, 2),
    records_processed INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    success BOOLEAN DEFAULT TRUE,
    error_count INTEGER DEFAULT 0,
    errors TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_job_runs_name_time ON scheduler_job_runs(job_name, start_time DESC);

COMMIT;

-- Summary
SELECT 'Schema created successfully!' AS status;
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN (
    'treasury_yields', 'macro_indicators', 'macro_indicator_values',
    'form4_filings', 'insider_sentiment', 'institutions',
    'form13f_filings', 'form13f_holdings', 'institutional_ownership',
    'short_interest', 'market_regime_history', 'current_market_regime',
    'regime_transitions', 'credit_risk_scores', 'current_credit_risk',
    'credit_risk_summary', 'scheduler_job_runs'
)
ORDER BY table_name;
