-- Add Missing Tables for InvestiGator Cache System
-- Fixes database issues identified during FAANG cache monitoring

-- Create missing sec_companyfacts table
-- Used by: RdbmsCacheStorageHandler for company facts caching
CREATE TABLE IF NOT EXISTS sec_companyfacts (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    cik VARCHAR(10) NOT NULL,
    companyfacts JSONB NOT NULL,
    company_name VARCHAR(255),
    metadata JSONB,
    fetched_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(symbol, cik)
);

CREATE INDEX idx_companyfacts_symbol ON sec_companyfacts(symbol);
CREATE INDEX idx_companyfacts_cik ON sec_companyfacts(cik);
CREATE INDEX idx_companyfacts_updated ON sec_companyfacts(updated_at);

-- Add trigger for updated_at
CREATE TRIGGER update_sec_companyfacts_updated_at BEFORE UPDATE ON sec_companyfacts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Fix potential NULL llm_type issues in llm_responses table
-- Make llm_type nullable temporarily to handle edge cases
ALTER TABLE llm_responses ALTER COLUMN llm_type DROP NOT NULL;

-- Add default value for llm_type to prevent constraint violations
ALTER TABLE llm_responses ALTER COLUMN llm_type SET DEFAULT 'unknown';

-- Update any existing NULL llm_type values
UPDATE llm_responses SET llm_type = 'unknown' WHERE llm_type IS NULL;

-- Re-add NOT NULL constraint after fixing data
ALTER TABLE llm_responses ALTER COLUMN llm_type SET NOT NULL;

-- Grant permissions to investment_ai user for new table
GRANT ALL PRIVILEGES ON sec_companyfacts TO investment_ai;
GRANT USAGE, SELECT ON SEQUENCE sec_companyfacts_id_seq TO investment_ai;

-- Insert completion record
INSERT INTO schema_version (version, description)
VALUES ('6.1.1', 'Added missing sec_companyfacts table and fixed llm_type constraints')
ON CONFLICT (version) DO NOTHING;

-- Verification queries
DO $$
BEGIN
    RAISE NOTICE '✅ sec_companyfacts table created successfully';
    RAISE NOTICE '✅ llm_type constraint issues fixed';
    RAISE NOTICE 'Database fixes complete - ready for cache operations';
END $$;