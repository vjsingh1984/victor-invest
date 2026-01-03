-- Migration: Add fundamental financial metrics to symbol table
-- Date: 2025-11-05
-- Purpose: Store absolute financial values (not per-share) for dynamic ratio calculation
--          With price + shares, can calculate: P/E = price / (net_income/shares), P/S = price / (revenue/shares)

ALTER TABLE symbol
ADD COLUMN IF NOT EXISTS revenue BIGINT,                    -- Total revenue (TTM)
ADD COLUMN IF NOT EXISTS net_income BIGINT,                 -- Net income (TTM)
ADD COLUMN IF NOT EXISTS total_assets BIGINT,               -- Total assets (most recent quarter)
ADD COLUMN IF NOT EXISTS total_liabilities BIGINT,          -- Total liabilities (most recent quarter)
ADD COLUMN IF NOT EXISTS stockholders_equity BIGINT,        -- Stockholders equity (most recent quarter)
ADD COLUMN IF NOT EXISTS operating_cash_flow BIGINT,        -- Operating cash flow (TTM)
ADD COLUMN IF NOT EXISTS free_cash_flow BIGINT,             -- Free cash flow (TTM)
ADD COLUMN IF NOT EXISTS gross_profit BIGINT,               -- Gross profit (TTM)
ADD COLUMN IF NOT EXISTS ebitda BIGINT,                     -- EBITDA (TTM)
ADD COLUMN IF NOT EXISTS total_debt BIGINT,                 -- Total debt (most recent quarter)
ADD COLUMN IF NOT EXISTS cash_and_equivalents BIGINT,       -- Cash and cash equivalents (most recent quarter)
ADD COLUMN IF NOT EXISTS dividends_paid BIGINT,             -- Dividends paid (TTM)
ADD COLUMN IF NOT EXISTS fiscal_period VARCHAR(10),         -- Fiscal period for metrics (e.g., '2025-Q3')
ADD COLUMN IF NOT EXISTS metrics_updated_at TIMESTAMP,      -- When metrics were last updated
ADD COLUMN IF NOT EXISTS metrics_source VARCHAR(50);        -- Source of metrics (e.g., 'sec_companyfacts', 'sec_bulk')

-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_symbol_revenue ON symbol(revenue);
CREATE INDEX IF NOT EXISTS idx_symbol_net_income ON symbol(net_income);
CREATE INDEX IF NOT EXISTS idx_symbol_mktcap_revenue ON symbol(mktcap, revenue);
CREATE INDEX IF NOT EXISTS idx_symbol_fiscal_period ON symbol(fiscal_period);

-- Add comments for documentation
COMMENT ON COLUMN symbol.revenue IS 'Total revenue (TTM) - not per share';
COMMENT ON COLUMN symbol.net_income IS 'Net income (TTM) - not per share. Use with shares to calc EPS: net_income/outstandingshares';
COMMENT ON COLUMN symbol.total_assets IS 'Total assets (most recent quarter)';
COMMENT ON COLUMN symbol.total_liabilities IS 'Total liabilities (most recent quarter)';
COMMENT ON COLUMN symbol.stockholders_equity IS 'Stockholders equity (most recent quarter)';
COMMENT ON COLUMN symbol.operating_cash_flow IS 'Operating cash flow (TTM)';
COMMENT ON COLUMN symbol.free_cash_flow IS 'Free cash flow (TTM)';
COMMENT ON COLUMN symbol.fiscal_period IS 'Fiscal period for these metrics (e.g., 2025-Q3)';

-- Example ratio calculations (for reference):
-- P/E Ratio: current_price / (net_income / outstandingshares)
-- P/S Ratio: current_price / (revenue / outstandingshares)
-- P/B Ratio: mktcap / stockholders_equity
-- Debt/Equity: total_debt / stockholders_equity
-- ROE: net_income / stockholders_equity
-- ROA: net_income / total_assets
-- Current Ratio: total_assets / total_liabilities (simplified)
