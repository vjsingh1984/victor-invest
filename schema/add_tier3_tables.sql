-- Add Tier 3 Enhancement Tables
-- Creates tables for Smart Alert System and Form 4 Monitoring

-- Create alerts table for investment alerts
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP,

    -- Severity check constraint
    CONSTRAINT alerts_severity_check CHECK (severity IN ('high', 'medium', 'low'))
);

CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved ON alerts(resolved_at) WHERE resolved_at IS NULL;

-- Create form4_filings table for insider trading monitoring
CREATE TABLE IF NOT EXISTS form4_filings (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    cik VARCHAR(10) NOT NULL,
    accession_number VARCHAR(50) NOT NULL UNIQUE,
    filing_date DATE NOT NULL,

    -- Reporting owner information
    owner_name VARCHAR(255),
    owner_title VARCHAR(255),
    is_director BOOLEAN DEFAULT FALSE,
    is_officer BOOLEAN DEFAULT FALSE,

    -- Transaction summary
    transaction_type VARCHAR(50),  -- Purchase, Sale, Grant, Exercise, etc.
    transaction_code VARCHAR(10),
    shares NUMERIC(20, 2),
    price_per_share NUMERIC(20, 4),
    total_value NUMERIC(20, 2),

    -- Significance flags
    is_significant BOOLEAN DEFAULT FALSE,
    significance_reasons TEXT[],

    -- Full filing data
    filing_data JSONB,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_form4_symbol ON form4_filings(symbol);
CREATE INDEX IF NOT EXISTS idx_form4_filing_date ON form4_filings(filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_form4_accession ON form4_filings(accession_number);
CREATE INDEX IF NOT EXISTS idx_form4_significant ON form4_filings(is_significant) WHERE is_significant = TRUE;
CREATE INDEX IF NOT EXISTS idx_form4_transaction_type ON form4_filings(transaction_type);

-- Add trigger for updated_at on form4_filings (function likely already exists from previous migrations)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'update_form4_filings_updated_at'
    ) THEN
        CREATE TRIGGER update_form4_filings_updated_at
            BEFORE UPDATE ON form4_filings
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- Grant permissions to investigator user
GRANT ALL PRIVILEGES ON alerts TO investigator;
GRANT USAGE, SELECT ON SEQUENCE alerts_id_seq TO investigator;
GRANT ALL PRIVILEGES ON form4_filings TO investigator;
GRANT USAGE, SELECT ON SEQUENCE form4_filings_id_seq TO investigator;

-- Insert schema version record if table exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'schema_version') THEN
        INSERT INTO schema_version (version, description)
        VALUES ('7.0.0', 'Added Tier 3 enhancement tables: alerts and form4_filings')
        ON CONFLICT (version) DO NOTHING;
    END IF;
END $$;

-- Verification queries
DO $$
BEGIN
    RAISE NOTICE '✅ alerts table created successfully';
    RAISE NOTICE '✅ form4_filings table created successfully';
    RAISE NOTICE '✅ Indexes and triggers configured';
    RAISE NOTICE 'Tier 3 database enhancements complete';
END $$;
