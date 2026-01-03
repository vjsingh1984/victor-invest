-- Migration: Expand sec_companyfacts_processed with additional canonical metrics
-- Date: 2025-11-05

ALTER TABLE sec_companyfacts_processed
    ADD COLUMN IF NOT EXISTS property_plant_equipment NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS accumulated_depreciation NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS property_plant_equipment_net NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS goodwill NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS intangible_assets NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS retained_earnings NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS research_and_development_expense NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS selling_general_administrative_expense NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS operating_expenses NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS interest_expense NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS income_tax_expense NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS effective_tax_rate NUMERIC(10, 4),
    ADD COLUMN IF NOT EXISTS interest_coverage NUMERIC(12, 4),
    ADD COLUMN IF NOT EXISTS depreciation_amortization NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS stock_based_compensation NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS deferred_revenue NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS accounts_payable NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS accrued_liabilities NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS preferred_stock_dividends NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS common_stock_dividends NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS treasury_stock NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS other_comprehensive_income NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS investing_cash_flow NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS financing_cash_flow NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS shares_outstanding NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS earnings_per_share NUMERIC(16, 4),
    ADD COLUMN IF NOT EXISTS earnings_per_share_diluted NUMERIC(16, 4),
    ADD COLUMN IF NOT EXISTS market_cap NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS book_value NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS book_value_per_share NUMERIC(16, 4),
    ADD COLUMN IF NOT EXISTS working_capital NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS net_debt NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS enterprise_value NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS dividend_payout_ratio NUMERIC(10, 4),
    ADD COLUMN IF NOT EXISTS dividend_yield NUMERIC(10, 4),
    ADD COLUMN IF NOT EXISTS return_on_assets NUMERIC(10, 4),
    ADD COLUMN IF NOT EXISTS return_on_equity NUMERIC(10, 4),
    ADD COLUMN IF NOT EXISTS asset_turnover NUMERIC(10, 4);

COMMENT ON COLUMN sec_companyfacts_processed.property_plant_equipment IS 'Gross property, plant, and equipment';
COMMENT ON COLUMN sec_companyfacts_processed.accumulated_depreciation IS 'Accumulated depreciation for PP&E';
COMMENT ON COLUMN sec_companyfacts_processed.property_plant_equipment_net IS 'Net PP&E (gross minus accumulated depreciation)';
COMMENT ON COLUMN sec_companyfacts_processed.goodwill IS 'Goodwill balance';
COMMENT ON COLUMN sec_companyfacts_processed.intangible_assets IS 'Intangible assets balance';
COMMENT ON COLUMN sec_companyfacts_processed.retained_earnings IS 'Retained earnings balance';
COMMENT ON COLUMN sec_companyfacts_processed.research_and_development_expense IS 'Research and development expense';
COMMENT ON COLUMN sec_companyfacts_processed.selling_general_administrative_expense IS 'Selling, general, and administrative expense';
COMMENT ON COLUMN sec_companyfacts_processed.operating_expenses IS 'Total operating expenses';
COMMENT ON COLUMN sec_companyfacts_processed.interest_expense IS 'Interest expense';
COMMENT ON COLUMN sec_companyfacts_processed.income_tax_expense IS 'Income tax expense';
COMMENT ON COLUMN sec_companyfacts_processed.effective_tax_rate IS 'Effective tax rate (percentage)';
COMMENT ON COLUMN sec_companyfacts_processed.interest_coverage IS 'Interest coverage ratio';
COMMENT ON COLUMN sec_companyfacts_processed.depreciation_amortization IS 'Depreciation and amortization expense';
COMMENT ON COLUMN sec_companyfacts_processed.stock_based_compensation IS 'Stock-based compensation expense';
COMMENT ON COLUMN sec_companyfacts_processed.deferred_revenue IS 'Deferred revenue balance';
COMMENT ON COLUMN sec_companyfacts_processed.accounts_payable IS 'Accounts payable balance';
COMMENT ON COLUMN sec_companyfacts_processed.accrued_liabilities IS 'Accrued liabilities balance';
COMMENT ON COLUMN sec_companyfacts_processed.preferred_stock_dividends IS 'Preferred stock dividends paid';
COMMENT ON COLUMN sec_companyfacts_processed.common_stock_dividends IS 'Common stock dividends paid';
COMMENT ON COLUMN sec_companyfacts_processed.treasury_stock IS 'Treasury stock balance';
COMMENT ON COLUMN sec_companyfacts_processed.other_comprehensive_income IS 'Accumulated other comprehensive income';
COMMENT ON COLUMN sec_companyfacts_processed.investing_cash_flow IS 'Net cash provided (used) by investing activities';
COMMENT ON COLUMN sec_companyfacts_processed.financing_cash_flow IS 'Net cash provided (used) by financing activities';
COMMENT ON COLUMN sec_companyfacts_processed.shares_outstanding IS 'Shares outstanding for the period';
COMMENT ON COLUMN sec_companyfacts_processed.earnings_per_share IS 'Basic EPS';
COMMENT ON COLUMN sec_companyfacts_processed.earnings_per_share_diluted IS 'Diluted EPS';
COMMENT ON COLUMN sec_companyfacts_processed.market_cap IS 'Market capitalization at period end';
COMMENT ON COLUMN sec_companyfacts_processed.book_value IS 'Book value of equity';
COMMENT ON COLUMN sec_companyfacts_processed.book_value_per_share IS 'Book value per share';
COMMENT ON COLUMN sec_companyfacts_processed.working_capital IS 'Working capital';
COMMENT ON COLUMN sec_companyfacts_processed.net_debt IS 'Net debt (total debt minus cash)';
COMMENT ON COLUMN sec_companyfacts_processed.enterprise_value IS 'Enterprise value';
COMMENT ON COLUMN sec_companyfacts_processed.dividend_payout_ratio IS 'Dividend payout ratio';
COMMENT ON COLUMN sec_companyfacts_processed.dividend_yield IS 'Dividend yield';
COMMENT ON COLUMN sec_companyfacts_processed.return_on_assets IS 'Return on assets (ratio)';
COMMENT ON COLUMN sec_companyfacts_processed.return_on_equity IS 'Return on equity (ratio)';
COMMENT ON COLUMN sec_companyfacts_processed.asset_turnover IS 'Asset turnover ratio';
