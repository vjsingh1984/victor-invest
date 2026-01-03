-- Phase 1: Schema Migration for Statement-Specific qtrs Tracking
-- Add columns to track qtrs for income statement and cash flow statement separately
-- Based on S&P 100 empirical analysis showing 80% mixed pattern

-- Add statement-specific qtrs columns
ALTER TABLE sec_companyfacts_processed
ADD COLUMN IF NOT EXISTS income_statement_qtrs SMALLINT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS cash_flow_statement_qtrs SMALLINT DEFAULT NULL;

-- Add comments for documentation
COMMENT ON COLUMN sec_companyfacts_processed.income_statement_qtrs IS
  'Duration in quarters for income statement metrics:
   1=individual quarter, 2=Q2 YTD, 3=Q3 YTD, 4=full year
   NULL=not determined yet or legacy data';

COMMENT ON COLUMN sec_companyfacts_processed.cash_flow_statement_qtrs IS
  'Duration in quarters for cash flow statement metrics:
   1=individual quarter, 2=Q2 YTD, 3=Q3 YTD, 4=full year
   NULL=not determined yet or legacy data';

-- Backfill existing data based on empirical S&P 100 pattern
-- Default to YTD pattern (80% of stocks) for safety
UPDATE sec_companyfacts_processed
SET
    income_statement_qtrs = CASE
        WHEN fiscal_period = 'Q1' THEN 1
        WHEN fiscal_period = 'Q2' THEN 2  -- Default to YTD (safe for 100% of stocks)
        WHEN fiscal_period = 'Q3' THEN 3  -- Default to YTD (safe for 100% of stocks)
        WHEN fiscal_period = 'FY' THEN 4
        ELSE NULL
    END,
    cash_flow_statement_qtrs = CASE
        WHEN fiscal_period = 'Q1' THEN 1
        WHEN fiscal_period = 'Q2' THEN 2  -- Always YTD for majority (80%)
        WHEN fiscal_period = 'Q3' THEN 3  -- Always YTD for majority (80%)
        WHEN fiscal_period = 'FY' THEN 4
        ELSE NULL
    END
WHERE income_statement_qtrs IS NULL OR cash_flow_statement_qtrs IS NULL;

-- Create index for common queries
CREATE INDEX IF NOT EXISTS idx_companyfacts_qtrs
ON sec_companyfacts_processed(
    symbol,
    fiscal_year,
    fiscal_period,
    income_statement_qtrs,
    cash_flow_statement_qtrs
);

-- Verify migration
SELECT
    fiscal_period,
    COUNT(*) as row_count,
    COUNT(DISTINCT income_statement_qtrs) as distinct_income_qtrs,
    COUNT(DISTINCT cash_flow_statement_qtrs) as distinct_cashflow_qtrs,
    STRING_AGG(DISTINCT income_statement_qtrs::TEXT, ', ' ORDER BY income_statement_qtrs::TEXT) as income_qtrs_values,
    STRING_AGG(DISTINCT cash_flow_statement_qtrs::TEXT, ', ' ORDER BY cash_flow_statement_qtrs::TEXT) as cashflow_qtrs_values
FROM sec_companyfacts_processed
GROUP BY fiscal_period
ORDER BY
    CASE fiscal_period
        WHEN 'Q1' THEN 1
        WHEN 'Q2' THEN 2
        WHEN 'Q3' THEN 3
        WHEN 'FY' THEN 4
    END;
