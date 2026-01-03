-- Migration 005: Add Regional Federal Reserve District Data Tables
-- Created: 2025-01-02
-- Description: Tables for storing regional Fed economic indicators from all 12 districts

-- Regional Fed Indicators (unified table for all districts)
CREATE TABLE IF NOT EXISTS regional_fed_indicators (
    id BIGSERIAL PRIMARY KEY,
    district VARCHAR(50) NOT NULL,  -- atlanta, philadelphia, chicago, etc.
    indicator_name VARCHAR(100) NOT NULL,  -- gdpnow, cfnai, manufacturing_survey, etc.
    observation_date DATE NOT NULL,
    indicator_data JSONB NOT NULL,  -- Full indicator data as JSON
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(district, indicator_name, observation_date)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_regional_fed_district ON regional_fed_indicators(district);
CREATE INDEX IF NOT EXISTS idx_regional_fed_indicator ON regional_fed_indicators(indicator_name);
CREATE INDEX IF NOT EXISTS idx_regional_fed_date ON regional_fed_indicators(observation_date DESC);
CREATE INDEX IF NOT EXISTS idx_regional_fed_district_date ON regional_fed_indicators(district, observation_date DESC);

-- GDPNow History (Atlanta Fed's nowcast)
CREATE TABLE IF NOT EXISTS gdpnow_history (
    id BIGSERIAL PRIMARY KEY,
    observation_date DATE NOT NULL,
    quarter VARCHAR(10) NOT NULL,  -- e.g., "2025Q1"
    gdp_estimate DECIMAL(5,2) NOT NULL,  -- GDP growth estimate
    previous_estimate DECIMAL(5,2),
    change_from_previous DECIMAL(5,2),
    blue_chip_consensus DECIMAL(5,2),
    outlook VARCHAR(50),  -- strong_contraction, contraction, weak_growth, etc.
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(observation_date, quarter)
);

CREATE INDEX IF NOT EXISTS idx_gdpnow_quarter ON gdpnow_history(quarter);
CREATE INDEX IF NOT EXISTS idx_gdpnow_date ON gdpnow_history(observation_date DESC);

-- CFNAI History (Chicago Fed National Activity Index)
CREATE TABLE IF NOT EXISTS cfnai_history (
    id BIGSERIAL PRIMARY KEY,
    observation_date DATE NOT NULL UNIQUE,
    cfnai DECIMAL(6,3) NOT NULL,  -- Chicago Fed National Activity Index
    cfnai_ma3 DECIMAL(6,3) NOT NULL,  -- 3-month moving average (key signal)
    production_income DECIMAL(6,3),
    employment DECIMAL(6,3),
    consumption_housing DECIMAL(6,3),
    sales_orders_inventories DECIMAL(6,3),
    condition VARCHAR(50),  -- recession, contraction_risk, below_trend, etc.
    recession_probability DECIMAL(5,2),  -- Implied probability
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cfnai_date ON cfnai_history(observation_date DESC);
CREATE INDEX IF NOT EXISTS idx_cfnai_condition ON cfnai_history(condition);

-- Financial Conditions Indexes
CREATE TABLE IF NOT EXISTS financial_conditions (
    id BIGSERIAL PRIMARY KEY,
    observation_date DATE NOT NULL,
    index_name VARCHAR(50) NOT NULL,  -- nfci, kcfsi, anfci
    index_value DECIMAL(8,4) NOT NULL,
    condition_level VARCHAR(50),  -- very_loose, loose, neutral, tight, etc.
    risk_subindex DECIMAL(8,4),
    credit_subindex DECIMAL(8,4),
    leverage_subindex DECIMAL(8,4),
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(observation_date, index_name)
);

CREATE INDEX IF NOT EXISTS idx_financial_conditions_name ON financial_conditions(index_name);
CREATE INDEX IF NOT EXISTS idx_financial_conditions_date ON financial_conditions(observation_date DESC);

-- Regional Manufacturing Surveys
CREATE TABLE IF NOT EXISTS regional_manufacturing_surveys (
    id BIGSERIAL PRIMARY KEY,
    observation_date DATE NOT NULL,
    survey_name VARCHAR(50) NOT NULL,  -- philly_bos, texas_mfg, kc_mfg, richmond_mfg, empire_state
    composite_index DECIMAL(6,2) NOT NULL,
    new_orders DECIMAL(6,2),
    shipments DECIMAL(6,2),
    employment DECIMAL(6,2),
    prices_paid DECIMAL(6,2),
    prices_received DECIMAL(6,2),
    future_activity DECIMAL(6,2),
    outlook VARCHAR(50),  -- strong_contraction, contraction, weak_expansion, etc.
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(observation_date, survey_name)
);

CREATE INDEX IF NOT EXISTS idx_regional_mfg_survey ON regional_manufacturing_surveys(survey_name);
CREATE INDEX IF NOT EXISTS idx_regional_mfg_date ON regional_manufacturing_surveys(observation_date DESC);

-- Inflation Expectations (Cleveland Fed and others)
CREATE TABLE IF NOT EXISTS inflation_expectations (
    id BIGSERIAL PRIMARY KEY,
    observation_date DATE NOT NULL,
    source VARCHAR(50) NOT NULL,  -- cleveland_fed, michigan, breakeven
    one_year DECIMAL(5,2),
    two_year DECIMAL(5,2),
    five_year DECIMAL(5,2),
    ten_year DECIMAL(5,2),
    five_year_five_year DECIMAL(5,2),  -- 5Y5Y forward
    inflation_risk_premium DECIMAL(5,2),
    outlook VARCHAR(50),
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(observation_date, source)
);

CREATE INDEX IF NOT EXISTS idx_inflation_exp_source ON inflation_expectations(source);
CREATE INDEX IF NOT EXISTS idx_inflation_exp_date ON inflation_expectations(observation_date DESC);

-- VIX and Volatility Data
CREATE TABLE IF NOT EXISTS volatility_indicators (
    id BIGSERIAL PRIMARY KEY,
    observation_date DATE NOT NULL UNIQUE,
    vix_spot DECIMAL(6,2) NOT NULL,
    vix_open DECIMAL(6,2),
    vix_high DECIMAL(6,2),
    vix_low DECIMAL(6,2),
    vix_change DECIMAL(6,2),
    vix_regime VARCHAR(50),  -- very_low, low, normal, elevated, high, extreme
    vix_front_month DECIMAL(6,2),
    vix_second_month DECIMAL(6,2),
    vix_term_structure VARCHAR(50),  -- contango, flat, backwardation
    skew_index DECIMAL(6,2),
    put_call_ratio DECIMAL(6,3),
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_volatility_date ON volatility_indicators(observation_date DESC);
CREATE INDEX IF NOT EXISTS idx_volatility_regime ON volatility_indicators(vix_regime);

-- Leading/Coincident Indexes by State
CREATE TABLE IF NOT EXISTS state_economic_indexes (
    id BIGSERIAL PRIMARY KEY,
    observation_date DATE NOT NULL,
    state_code VARCHAR(2) NOT NULL,  -- US for national
    index_type VARCHAR(20) NOT NULL,  -- leading, coincident
    index_value DECIMAL(8,3) NOT NULL,
    one_month_change DECIMAL(6,3),
    three_month_change DECIMAL(6,3),
    twelve_month_change DECIMAL(6,3),
    interpretation VARCHAR(100),
    source_hash VARCHAR(64),
    source_fetch_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(observation_date, state_code, index_type)
);

CREATE INDEX IF NOT EXISTS idx_state_econ_state ON state_economic_indexes(state_code);
CREATE INDEX IF NOT EXISTS idx_state_econ_type ON state_economic_indexes(index_type);
CREATE INDEX IF NOT EXISTS idx_state_econ_date ON state_economic_indexes(observation_date DESC);

-- Watermarks for incremental collection
CREATE TABLE IF NOT EXISTS regional_fed_watermarks (
    district VARCHAR(50) PRIMARY KEY,
    last_observation_date DATE,
    last_fetch_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Comments for documentation
COMMENT ON TABLE regional_fed_indicators IS 'Unified storage for all regional Fed economic indicators';
COMMENT ON TABLE gdpnow_history IS 'Atlanta Fed GDPNow real-time GDP estimates';
COMMENT ON TABLE cfnai_history IS 'Chicago Fed National Activity Index - key recession indicator';
COMMENT ON TABLE financial_conditions IS 'Financial conditions indexes (NFCI, KCFSI)';
COMMENT ON TABLE regional_manufacturing_surveys IS 'Regional Fed manufacturing surveys';
COMMENT ON TABLE inflation_expectations IS 'Model-based inflation expectations';
COMMENT ON TABLE volatility_indicators IS 'VIX and volatility market data';
COMMENT ON TABLE state_economic_indexes IS 'State-level leading and coincident indexes';
