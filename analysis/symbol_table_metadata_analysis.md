# Symbol Table Metadata Analysis

**Analysis Date**: 2025-11-07
**Total Symbols**: 25,322 (17,013 stocks, 1,233 ETFs)

## Table of Contents
1. [Schema Overview](#schema-overview)
2. [Data Coverage Analysis](#data-coverage-analysis)
3. [Data Quality Issues](#data-quality-issues)
4. [Recommendations](#recommendations)
5. [Missing Metadata](#missing-metadata)

---

## Schema Overview

### Column Categories

**Identity & Classification** (8 columns):
- `ticker` (PK) - Symbol identifier ✅ 100% coverage
- `description` - Company/fund name (36.5% coverage)
- `isstock` / `isetf` - Type flags (67.2% stocks, 4.9% ETFs)
- `exchange` - Trading venue (0.7% coverage) ⚠️
- `Country` - Domicile (28.5% coverage)
- `stockid` / `id` - Internal IDs

**Sector Classification** (6 columns):
- `Sector` / `Industry` - Yahoo Finance (28.1% coverage)
- `sec_sector` / `sec_industry` - Canonical GICS (28.2% coverage)
- `sic_code` - SIC code (0% coverage) ❌
- `cik` - SEC Central Index Key (26.1% coverage)

**Market Data** (4 columns):
- `mktcap` - Market capitalization (34.2% coverage)
- `outstandingshares` - Share count (6.1% coverage) ⚠️
- `ipo` - IPO year
- `index` - Index membership (sparse)

**Financial Metrics** (14 columns from symbol_update agent):
- Revenue, net_income, total_assets, total_liabilities
- stockholders_equity, operating_cash_flow, free_cash_flow
- gross_profit, ebitda, total_debt, cash_and_equivalents
- dividends_paid, fiscal_period, metrics_updated_at
- **Coverage**: 0% (not yet populated) ❌

**Valuation Models** (22 columns):
- fair_value_* (7 models: blended, dcf, ggm, ps, pe, pb, ev_ebitda)
- Model metadata: agreement_score, confidence, applicable_models
- Rule of 40: score, classification
- Ratios: revenue_growth_rate, fcf_margin, ps_ratio, pe_ratio, pb_ratio, peg_ratio, ev_ebitda_ratio
- Sector comparisons: sector_median_ps, ps_premium_discount
- DCF parameters: wacc, terminal_growth_rate, dcf_projection_years
- **Coverage**: 0% (not yet populated) ❌

**Beta/Correlation** (14 columns):
- `b_1_month`, `b_3_month`, `b_6_month`, `b_12_month`, `b_24_month`, `b_36_month`, `b_60_month`
- `r2_1_month`, `r2_3_month`, `r2_6_month`, `r2_12_month`, `r2_24_month`, `r2_36_month`, `r2_60_month`
- **Coverage**: Good for tier companies (populated with market correlation data)

**Control Flags** (6 columns):
- `islisted`, `skiptametric`, `queryibkrapi`, `ballmetrics`, `ismlenabled`
- `divergence_flag` - Valuation model divergence indicator
- **Coverage**: Defaults set, functional

**Timestamps** (3 columns):
- `lastupdts` - Last metadata update
- `metrics_updated_at` - Last financial metrics update
- `valuation_updated_at` - Last valuation update

**JSON Storage** (1 column):
- `valuation_models_json` - Full model details (JSONB)

---

## Data Coverage Analysis

### Coverage by Symbol Type

| Metric | Total | Stocks | ETFs | Coverage % |
|--------|-------|--------|------|------------|
| **Total Symbols** | 25,322 | 17,013 | 1,233 | - |
| Has Description | 9,236 | 8,012 | 1,225 | 36.5% |
| Has Country | 7,217 | 7,217 | 1 | 28.5% |
| Has Exchange | 172 | 78 | 94 | 0.7% ⚠️ |
| Has Sector (Yahoo) | 7,129 | 7,129 | 1 | 28.1% |
| Has sec_sector | 7,135 | 7,135 | 1 | 28.2% |
| Has sec_industry | 7,129 | 7,129 | 1 | 28.1% |
| Has CIK | 6,604 | 5,703 | 79 | 26.1% |
| Has SIC Code | 0 | 0 | 0 | 0% ❌ |
| Has Market Cap | 8,667 | 7,496 | 1,172 | 34.2% |
| Has Outstanding Shares | 1,542 | 1,542 | 1 | 6.1% ⚠️ |
| Has Financials | 0 | 0 | 0 | 0% ❌ |
| Has Fair Value | 0 | 0 | 0 | 0% ❌ |

### Categorical Data Distribution

**Exchange** (only 172 symbols have this):
- NASDAQ: 134 symbols
- NYSE: 36 symbols
- TSE: 1 symbol
- Empty string: 1 symbol
- **Issue**: 99.3% missing exchange data

**Country** (7,217 symbols):
- United States: 5,932 (82.2%)
- China: 211 (2.9%)
- Canada: 201 (2.8%)
- Israel: 127 (1.8%)
- United Kingdom: 98 (1.4%)
- Other: 648 (9.0%)

**Sectors** (11 canonical + 1 miscellaneous):
- Financials: 1,830
- Healthcare: 1,280
- Industrials: 1,154
- Consumer Discretionary: 1,062
- Technology: 861
- Real Estate: 287
- Energy: 184
- Utilities: 177
- Consumer Staples: 145
- Communication Services: 64
- Materials: 45
- Miscellaneous: 46

**Industries**: 163 unique values (well-distributed)

---

## Data Quality Issues

### Critical Issues (High Impact)

1. **SIC Code - 100% Missing** ❌
   - No SIC codes populated
   - Impact: Cannot classify by SIC industry groups
   - **Recommendation**: Fetch from SEC EDGAR or use CIK mapping

2. **Exchange - 99.3% Missing** ❌
   - Only 172/25,322 symbols have exchange data
   - Impact: Cannot filter by exchange, analyze venue-specific patterns
   - **Recommendation**: Populate from Yahoo Finance or SEC filings

3. **Financial Metrics - 100% Missing** ❌
   - Columns exist but unpopulated: revenue, net_income, total_assets, etc.
   - Impact: symbol_update agent not yet run
   - **Recommendation**: Run symbol_update agent for analyzed symbols

4. **Valuation Models - 100% Missing** ❌
   - Fair value columns unpopulated
   - Impact: Cannot compare intrinsic value to market price
   - **Recommendation**: Populate after financial metrics available

### Medium Issues

5. **Outstanding Shares - 93.9% Missing** ⚠️
   - Only 1,542/25,322 symbols have share count
   - Impact: Cannot calculate per-share metrics (EPS, BVPS, etc.)
   - **Recommendation**: Fetch from SEC filings or financial data APIs

6. **Country - 71.5% Missing** ⚠️
   - 18,105 symbols missing country data
   - Impact: Cannot analyze by geography, apply country-specific models
   - **Recommendation**: Populate from SEC filings (for US stocks), Yahoo Finance

7. **Market Cap - 65.8% Missing** ⚠️
   - 16,655 symbols missing market cap
   - Impact: Cannot classify by size (large/mid/small cap)
   - **Recommendation**: Calculate as price × outstanding shares, or fetch from APIs

8. **Sector/Industry - 71.8% Missing** ⚠️
   - Despite recent backfill (7,135 symbols), still 18,187 missing
   - Impact: Cannot apply sector-specific valuation models
   - **Recommendation**:
     - Extend backfill to more symbols
     - Use SEC SIC code → GICS sector mapping
     - Fetch from additional data sources

### Low Issues

9. **Description - 63.5% Missing**
   - 16,086 symbols without company/fund names
   - Impact: User experience (harder to identify symbols)
   - **Recommendation**: Fetch from SEC filings, Yahoo Finance, or exchange APIs

10. **IPO Year - Not Analyzed**
    - Coverage unknown
    - Impact: Cannot analyze by company age/maturity
    - **Recommendation**: Fetch from SEC filings or financial databases

11. **Index Membership - Very Sparse**
    - Only ~30 symbols have index data
    - Impact: Cannot identify index constituents (S&P 500, Russell 2000, etc.)
    - **Recommendation**:
      - Scrape index constituent lists (S&P, Russell, Dow, NASDAQ-100)
      - Add `sp500`, `sp400`, `sp600`, `russell1000`, `russell2000`, `nasdaq100` boolean flags

---

## Recommendations

### Immediate Actions (High Priority)

1. **Populate Exchange Data**
   ```sql
   -- Add new data source or fetch from Yahoo Finance
   -- Target: 95%+ coverage for active stocks/ETFs
   ```

2. **Populate SIC Codes**
   ```sql
   -- Use SEC EDGAR CIK → SIC mapping
   -- Or derive from existing CIK data
   UPDATE symbol s
   SET sic_code = (
       SELECT sic FROM sec_company_info
       WHERE cik = s.cik
   )
   WHERE s.cik IS NOT NULL;
   ```

3. **Run Symbol Update Agent**
   - Execute for symbols with existing fundamental analysis
   - Populates financial metrics (revenue, income, assets, etc.)

4. **Extend Sector/Industry Coverage**
   - Backfill using additional sources:
     - SEC SIC → GICS mapping
     - Yahoo Finance bulk API
     - Manual curated lists for major indices

### Schema Enhancements (Medium Priority)

5. **Add Index Membership Flags**
   ```sql
   ALTER TABLE symbol
   ADD COLUMN sp500 BOOLEAN DEFAULT FALSE,
   ADD COLUMN sp400 BOOLEAN DEFAULT FALSE,
   ADD COLUMN sp600 BOOLEAN DEFAULT FALSE,
   ADD COLUMN russell1000 BOOLEAN DEFAULT FALSE,
   ADD COLUMN russell2000 BOOLEAN DEFAULT FALSE,
   ADD COLUMN russell3000 BOOLEAN DEFAULT FALSE,
   ADD COLUMN nasdaq100 BOOLEAN DEFAULT FALSE,
   ADD COLUMN dow30 BOOLEAN DEFAULT FALSE;
   ```

6. **Add Company Size Classification**
   ```sql
   ALTER TABLE symbol
   ADD COLUMN market_cap_tier VARCHAR(20), -- 'mega', 'large', 'mid', 'small', 'micro', 'nano'
   ADD COLUMN market_cap_updated_at TIMESTAMP;
   ```

7. **Add Data Quality Metadata**
   ```sql
   ALTER TABLE symbol
   ADD COLUMN data_quality_score NUMERIC, -- 0-100 score based on completeness
   ADD COLUMN data_sources JSONB, -- Track which fields came from which sources
   ADD COLUMN last_verified_at TIMESTAMP;
   ```

8. **Add ESG Data** (if applicable)
   ```sql
   ALTER TABLE symbol
   ADD COLUMN esg_score NUMERIC,
   ADD COLUMN esg_controversy_score NUMERIC,
   ADD COLUMN esg_updated_at TIMESTAMP;
   ```

9. **Add Dividend/Yield Data**
   ```sql
   ALTER TABLE symbol
   ADD COLUMN dividend_yield NUMERIC,
   ADD COLUMN dividend_frequency VARCHAR(20), -- 'annual', 'quarterly', 'monthly'
   ADD COLUMN ex_dividend_date DATE,
   ADD COLUMN dividend_growth_5y NUMERIC;
   ```

10. **Add Analyst Coverage**
    ```sql
    ALTER TABLE symbol
    ADD COLUMN analyst_count INTEGER,
    ADD COLUMN consensus_rating VARCHAR(20), -- 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    ADD COLUMN price_target_mean NUMERIC,
    ADD COLUMN price_target_low NUMERIC,
    ADD COLUMN price_target_high NUMERIC,
    ADD COLUMN analyst_updated_at TIMESTAMP;
    ```

### Data Normalization (Low Priority)

11. **Normalize Column Naming**
    - Inconsistent case: `Country`, `Sector`, `Industry` (PascalCase) vs `ticker`, `description` (lowercase)
    - **Recommendation**: Standardize to snake_case
      - `Country` → `country`
      - `Sector` → `sector_yahoo`
      - `Industry` → `industry_yahoo`

12. **Add Constraints**
    ```sql
    -- Ensure ticker is unique and not empty
    ALTER TABLE symbol ADD CONSTRAINT ticker_not_empty CHECK (ticker != '');

    -- Ensure market cap tier matches market cap value
    -- (needs trigger or application logic)

    -- Ensure at least one of isstock/isetf is true
    ALTER TABLE symbol ADD CONSTRAINT type_specified
    CHECK (isstock = true OR isetf = true);
    ```

---

## Missing Metadata Categories

### Not Currently Tracked (Consider Adding)

1. **Trading Characteristics**
   - Average daily volume (10d, 30d, 90d)
   - Bid-ask spread
   - Short interest %
   - Institutional ownership %
   - Insider ownership %

2. **Options Data**
   - Options available (boolean)
   - Implied volatility (30d, 60d, 90d)
   - Options volume

3. **Credit Ratings** (for bonds/corporate debt)
   - S&P rating
   - Moody's rating
   - Fitch rating

4. **Peer Comparisons**
   - Peer group identifiers (FK to peer_group table)
   - Rank within peer group (by market cap, revenue, etc.)

5. **Historical Milestones**
   - All-time high price + date
   - All-time low price + date
   - 52-week high/low

6. **Fundamental Indicators**
   - Altman Z-Score (bankruptcy risk)
   - Piotroski F-Score (financial strength)
   - Magic Formula rank

7. **Momentum Indicators**
   - RSI (14-day, 50-day)
   - MACD signal
   - 50-day / 200-day SMA crossover status

---

## Summary Statistics

### Coverage Scorecard

| Category | Coverage | Grade |
|----------|----------|-------|
| Identity (ticker, description) | 36.5% | D |
| Classification (sector, industry) | 28.2% | F |
| Geography (country, exchange) | 28.5% / 0.7% | F / F |
| Market Data (mktcap, shares) | 34.2% / 6.1% | D / F |
| SEC Data (CIK, SIC) | 26.1% / 0% | F / F |
| Financial Metrics | 0% | F |
| Valuation Models | 0% | F |
| Beta/Correlation | Good | B |
| Control Flags | 100% | A |

**Overall Data Quality: D- (30% weighted average)**

### Recommended Priority Order

1. **Phase 1 (Critical)**: Exchange, SIC Code, Financial Metrics
2. **Phase 2 (High)**: Outstanding Shares, Market Cap, Extended Sector Coverage
3. **Phase 3 (Medium)**: Index Flags, Country Backfill, Description
4. **Phase 4 (Low)**: Column naming, ESG, Dividends, Analyst Coverage
5. **Phase 5 (Future)**: Trading characteristics, options data, peer comparisons

---

## Tier Companies - Current State

| Ticker | Description | Country | Exchange | Sector | Industry | CIK | SIC | MktCap | Beta | Financials |
|--------|-------------|---------|----------|--------|----------|-----|-----|--------|------|------------|
| AAPL | ✅ | ✅ | ❌ | ✅ Technology | ✅ Consumer Electronics | ✅ | ❌ | ✅ | ✅ | ❌ |
| MSFT | ✅ | ✅ | ❌ | ✅ Technology | ✅ Software | ✅ | ❌ | ✅ | ✅ | ❌ |
| NVDA | ✅ | ✅ | ❌ | ✅ Technology | ✅ Semiconductors | ✅ | ❌ | ✅ | ✅ | ❌ |
| JNJ | ✅ | ✅ | ❌ | ✅ Healthcare | ✅ Pharma | ✅ | ❌ | ✅ | ✅ | ❌ |
| XOM | ✅ | ✅ | ❌ | ✅ Energy | ✅ Oil & Gas | ✅ | ❌ | ✅ | ✅ | ❌ |
| NEE | ✅ | ✅ | ❌ | ✅ Utilities | ✅ Electric | ✅ | ❌ | ✅ | ✅ | ❌ |

**Tier Companies Coverage**: 71% (5/7 categories populated)
