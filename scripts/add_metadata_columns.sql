-- Add missing metadata columns to symbol table
-- Run this migration to enhance the schema

-- Index membership flags
ALTER TABLE symbol
ADD COLUMN IF NOT EXISTS sp500 BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS sp400 BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS sp600 BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS russell1000 BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS russell2000 BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS russell3000 BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS nasdaq100 BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS dow30 BOOLEAN DEFAULT FALSE;

-- Market cap tier classification
ALTER TABLE symbol
ADD COLUMN IF NOT EXISTS market_cap_tier VARCHAR(20), -- 'mega', 'large', 'mid', 'small', 'micro', 'nano'
ADD COLUMN IF NOT EXISTS market_cap_updated_at TIMESTAMP;

-- Data quality tracking
ALTER TABLE symbol
ADD COLUMN IF NOT EXISTS data_quality_score NUMERIC, -- 0-100 score
ADD COLUMN IF NOT EXISTS data_completeness_pct NUMERIC, -- Percentage of non-null fields
ADD COLUMN IF NOT EXISTS last_data_refresh TIMESTAMP;

-- Dividend data
ALTER TABLE symbol
ADD COLUMN IF NOT EXISTS dividend_yield NUMERIC,
ADD COLUMN IF NOT EXISTS annual_dividend NUMERIC,
ADD COLUMN IF NOT EXISTS ex_dividend_date DATE,
ADD COLUMN IF NOT EXISTS dividend_frequency VARCHAR(20); -- 'annual', 'quarterly', 'monthly', 'none'

-- Trading metrics
ALTER TABLE symbol
ADD COLUMN IF NOT EXISTS avg_volume_30d BIGINT,
ADD COLUMN IF NOT EXISTS avg_volume_90d BIGINT,
ADD COLUMN IF NOT EXISTS float_shares BIGINT,
ADD COLUMN IF NOT EXISTS short_interest_pct NUMERIC,
ADD COLUMN IF NOT EXISTS institutional_ownership_pct NUMERIC;

-- Price metrics
ALTER TABLE symbol
ADD COLUMN IF NOT EXISTS current_price NUMERIC,
ADD COLUMN IF NOT EXISTS high_52week NUMERIC,
ADD COLUMN IF NOT EXISTS low_52week NUMERIC,
ADD COLUMN IF NOT EXISTS price_updated_at TIMESTAMP;

-- Analyst coverage
ALTER TABLE symbol
ADD COLUMN IF NOT EXISTS analyst_count INTEGER,
ADD COLUMN IF NOT EXISTS consensus_rating VARCHAR(20), -- 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
ADD COLUMN IF NOT EXISTS price_target_mean NUMERIC,
ADD COLUMN IF NOT EXISTS price_target_low NUMERIC,
ADD COLUMN IF NOT EXISTS price_target_high NUMERIC;

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_symbol_sp500 ON symbol(sp500) WHERE sp500 = true;
CREATE INDEX IF NOT EXISTS idx_symbol_nasdaq100 ON symbol(nasdaq100) WHERE nasdaq100 = true;
CREATE INDEX IF NOT EXISTS idx_symbol_dow30 ON symbol(dow30) WHERE dow30 = true;
CREATE INDEX IF NOT EXISTS idx_symbol_market_cap_tier ON symbol(market_cap_tier);
CREATE INDEX IF NOT EXISTS idx_symbol_sic_code ON symbol(sic_code);

-- Add comments
COMMENT ON COLUMN symbol.sp500 IS 'S&P 500 index membership';
COMMENT ON COLUMN symbol.nasdaq100 IS 'NASDAQ-100 index membership';
COMMENT ON COLUMN symbol.dow30 IS 'Dow Jones Industrial Average membership';
COMMENT ON COLUMN symbol.market_cap_tier IS 'Market capitalization tier: mega (>$200B), large ($10B-$200B), mid ($2B-$10B), small ($300M-$2B), micro ($50M-$300M), nano (<$50M)';
COMMENT ON COLUMN symbol.data_quality_score IS 'Data quality score 0-100 based on completeness and freshness';
COMMENT ON COLUMN symbol.dividend_yield IS 'Annual dividend yield as percentage';
COMMENT ON COLUMN symbol.analyst_count IS 'Number of analysts covering this symbol';
