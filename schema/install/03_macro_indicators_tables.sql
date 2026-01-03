-- InvestiGator Macro Indicators Tables
-- Version: 7.0.0
-- RDBMS-Agnostic: Works with PostgreSQL and SQLite
-- Copyright (c) 2025 Vijaykumar Singh
-- Licensed under Apache License 2.0
--
-- Tables for: treasury_yields, macro_indicators, regional Fed data, VIX

-- ============================================================================
-- TREASURY YIELDS
-- ============================================================================

CREATE TABLE IF NOT EXISTS treasury_yields (
    date TEXT PRIMARY KEY,
    yield_1m REAL,
    yield_3m REAL,
    yield_6m REAL,
    yield_1y REAL,
    yield_2y REAL,
    yield_5y REAL,
    yield_10y REAL,
    yield_20y REAL,
    yield_30y REAL,
    spread_10y_2y REAL,
    spread_10y_3m REAL,
    is_inverted INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_treasury_yields_date ON treasury_yields(date DESC);

-- ============================================================================
-- MACRO INDICATORS (Series metadata)
-- ============================================================================

CREATE TABLE IF NOT EXISTS macro_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT UNIQUE NOT NULL,
    name TEXT,
    category TEXT,
    frequency TEXT,
    units TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_macro_indicators_series ON macro_indicators(series_id);
CREATE INDEX IF NOT EXISTS idx_macro_indicators_category ON macro_indicators(category);

-- ============================================================================
-- MACRO INDICATOR VALUES
-- ============================================================================

CREATE TABLE IF NOT EXISTS macro_indicator_values (
    indicator_id INTEGER REFERENCES macro_indicators(id),
    date TEXT NOT NULL,
    value REAL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (indicator_id, date)
);

CREATE INDEX IF NOT EXISTS idx_macro_values_date ON macro_indicator_values(date DESC);

-- ============================================================================
-- REGIONAL FED INDICATORS (Unified)
-- ============================================================================

CREATE TABLE IF NOT EXISTS regional_fed_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    district TEXT NOT NULL,
    indicator_name TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    indicator_data TEXT NOT NULL,  -- JSON
    source_hash TEXT,
    source_fetch_timestamp TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(district, indicator_name, observation_date)
);

CREATE INDEX IF NOT EXISTS idx_regional_fed_district ON regional_fed_indicators(district);
CREATE INDEX IF NOT EXISTS idx_regional_fed_indicator ON regional_fed_indicators(indicator_name);
CREATE INDEX IF NOT EXISTS idx_regional_fed_date ON regional_fed_indicators(observation_date DESC);

-- ============================================================================
-- GDPNOW HISTORY (Atlanta Fed)
-- ============================================================================

CREATE TABLE IF NOT EXISTS gdpnow_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_date TEXT NOT NULL,
    quarter TEXT NOT NULL,
    gdp_estimate REAL NOT NULL,
    previous_estimate REAL,
    change_from_previous REAL,
    blue_chip_consensus REAL,
    outlook TEXT,
    source_hash TEXT,
    source_fetch_timestamp TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(observation_date, quarter)
);

CREATE INDEX IF NOT EXISTS idx_gdpnow_quarter ON gdpnow_history(quarter);
CREATE INDEX IF NOT EXISTS idx_gdpnow_date ON gdpnow_history(observation_date DESC);

-- ============================================================================
-- CFNAI HISTORY (Chicago Fed National Activity Index)
-- ============================================================================

CREATE TABLE IF NOT EXISTS cfnai_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_date TEXT NOT NULL UNIQUE,
    cfnai REAL NOT NULL,
    cfnai_ma3 REAL NOT NULL,
    production_income REAL,
    employment REAL,
    consumption_housing REAL,
    sales_orders_inventories REAL,
    condition TEXT,
    recession_probability REAL,
    source_hash TEXT,
    source_fetch_timestamp TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cfnai_date ON cfnai_history(observation_date DESC);
CREATE INDEX IF NOT EXISTS idx_cfnai_condition ON cfnai_history(condition);

-- ============================================================================
-- FINANCIAL CONDITIONS INDEXES (NFCI, KCFSI)
-- ============================================================================

CREATE TABLE IF NOT EXISTS financial_conditions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_date TEXT NOT NULL,
    index_name TEXT NOT NULL,
    index_value REAL NOT NULL,
    condition_level TEXT,
    risk_subindex REAL,
    credit_subindex REAL,
    leverage_subindex REAL,
    source_hash TEXT,
    source_fetch_timestamp TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(observation_date, index_name)
);

CREATE INDEX IF NOT EXISTS idx_financial_conditions_name ON financial_conditions(index_name);
CREATE INDEX IF NOT EXISTS idx_financial_conditions_date ON financial_conditions(observation_date DESC);

-- ============================================================================
-- REGIONAL MANUFACTURING SURVEYS
-- ============================================================================

CREATE TABLE IF NOT EXISTS regional_manufacturing_surveys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_date TEXT NOT NULL,
    survey_name TEXT NOT NULL,
    composite_index REAL NOT NULL,
    new_orders REAL,
    shipments REAL,
    employment REAL,
    prices_paid REAL,
    prices_received REAL,
    future_activity REAL,
    outlook TEXT,
    source_hash TEXT,
    source_fetch_timestamp TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(observation_date, survey_name)
);

CREATE INDEX IF NOT EXISTS idx_regional_mfg_survey ON regional_manufacturing_surveys(survey_name);
CREATE INDEX IF NOT EXISTS idx_regional_mfg_date ON regional_manufacturing_surveys(observation_date DESC);

-- ============================================================================
-- INFLATION EXPECTATIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS inflation_expectations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_date TEXT NOT NULL,
    source TEXT NOT NULL,
    one_year REAL,
    two_year REAL,
    five_year REAL,
    ten_year REAL,
    five_year_five_year REAL,
    inflation_risk_premium REAL,
    outlook TEXT,
    source_hash TEXT,
    source_fetch_timestamp TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(observation_date, source)
);

CREATE INDEX IF NOT EXISTS idx_inflation_exp_source ON inflation_expectations(source);
CREATE INDEX IF NOT EXISTS idx_inflation_exp_date ON inflation_expectations(observation_date DESC);

-- ============================================================================
-- VOLATILITY INDICATORS (VIX, SKEW)
-- ============================================================================

CREATE TABLE IF NOT EXISTS volatility_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_date TEXT NOT NULL UNIQUE,
    vix_spot REAL NOT NULL,
    vix_open REAL,
    vix_high REAL,
    vix_low REAL,
    vix_change REAL,
    vix_regime TEXT,
    vix_front_month REAL,
    vix_second_month REAL,
    vix_term_structure TEXT,
    skew_index REAL,
    put_call_ratio REAL,
    source_hash TEXT,
    source_fetch_timestamp TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_volatility_date ON volatility_indicators(observation_date DESC);
CREATE INDEX IF NOT EXISTS idx_volatility_regime ON volatility_indicators(vix_regime);

-- ============================================================================
-- MARKET REGIME
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_regime_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT UNIQUE NOT NULL,
    regime TEXT,
    credit_cycle_phase TEXT,
    volatility_regime TEXT,
    recession_probability REAL,
    yield_curve_inverted INTEGER DEFAULT 0,
    vix_level REAL,
    credit_spread REAL,
    risk_off_signal INTEGER DEFAULT 0,
    recommendations TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_market_regime_date ON market_regime_history(snapshot_date DESC);

CREATE TABLE IF NOT EXISTS current_market_regime (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    regime TEXT,
    credit_cycle_phase TEXT,
    volatility_regime TEXT,
    recession_probability REAL,
    yield_curve_inverted INTEGER DEFAULT 0,
    risk_off_signal INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT (datetime('now'))
);

-- ============================================================================
-- STATE ECONOMIC INDEXES
-- ============================================================================

CREATE TABLE IF NOT EXISTS state_economic_indexes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_date TEXT NOT NULL,
    state_code TEXT NOT NULL,
    index_type TEXT NOT NULL,
    index_value REAL NOT NULL,
    one_month_change REAL,
    three_month_change REAL,
    twelve_month_change REAL,
    interpretation TEXT,
    source_hash TEXT,
    source_fetch_timestamp TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(observation_date, state_code, index_type)
);

CREATE INDEX IF NOT EXISTS idx_state_econ_state ON state_economic_indexes(state_code);
CREATE INDEX IF NOT EXISTS idx_state_econ_type ON state_economic_indexes(index_type);
CREATE INDEX IF NOT EXISTS idx_state_econ_date ON state_economic_indexes(observation_date DESC);

-- Insert version
INSERT OR IGNORE INTO schema_version (version, description)
VALUES ('7.0.3', 'Macro indicators tables - treasury, Fed, VIX, market regime');
