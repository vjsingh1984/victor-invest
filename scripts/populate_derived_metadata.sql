-- Populate derived metadata from existing columns
-- This script calculates and populates values that can be derived from existing data

-- 1. Calculate and populate market cap tiers based on mktcap
UPDATE symbol
SET market_cap_tier = CASE
    WHEN mktcap >= 200000000000 THEN 'mega'         -- >= $200B
    WHEN mktcap >= 10000000000 THEN 'large'         -- $10B - $200B
    WHEN mktcap >= 2000000000 THEN 'mid'            -- $2B - $10B
    WHEN mktcap >= 300000000 THEN 'small'           -- $300M - $2B
    WHEN mktcap >= 50000000 THEN 'micro'            -- $50M - $300M
    WHEN mktcap > 0 THEN 'nano'                     -- < $50M
    ELSE NULL
END,
market_cap_updated_at = NOW()
WHERE mktcap IS NOT NULL AND mktcap > 0;

-- 2. Calculate data completeness percentage
-- Count non-null values for key metadata fields
UPDATE symbol
SET data_completeness_pct = (
    SELECT ROUND(100.0 * (
        CASE WHEN ticker IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN description IS NOT NULL AND description != '' THEN 1 ELSE 0 END +
        CASE WHEN exchange IS NOT NULL AND exchange != '' THEN 1 ELSE 0 END +
        CASE WHEN "Country" IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN sec_sector IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN sec_industry IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN cik IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN sic_code IS NOT NULL AND sic_code > 0 THEN 1 ELSE 0 END +
        CASE WHEN mktcap IS NOT NULL AND mktcap > 0 THEN 1 ELSE 0 END +
        CASE WHEN outstandingshares IS NOT NULL THEN 1 ELSE 0 END
    ) / 10.0, 1)
),
last_data_refresh = NOW();

-- 3. Calculate data quality score (0-100)
-- Based on completeness, freshness, and data source reliability
UPDATE symbol
SET data_quality_score = (
    -- Completeness (60% weight)
    COALESCE(data_completeness_pct * 0.6, 0) +

    -- Has key identifiers (20% weight)
    CASE
        WHEN cik IS NOT NULL AND sec_sector IS NOT NULL THEN 20
        WHEN cik IS NOT NULL OR sec_sector IS NOT NULL THEN 10
        ELSE 0
    END +

    -- Freshness (20% weight)
    CASE
        WHEN lastupdts >= NOW() - INTERVAL '30 days' THEN 20
        WHEN lastupdts >= NOW() - INTERVAL '90 days' THEN 10
        WHEN lastupdts >= NOW() - INTERVAL '180 days' THEN 5
        ELSE 0
    END
);

-- 4. Mark symbols with comprehensive data (quality score >= 70)
-- These can be prioritized for analysis

-- 5. Update tier companies with known index memberships
-- S&P 500 (sample - would need full list)
UPDATE symbol
SET sp500 = true
WHERE ticker IN (
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    'BRK.B', 'UNH', 'JNJ', 'JPM', 'V', 'XOM', 'PG', 'MA', 'HD', 'CVX',
    'LLY', 'ABBV', 'MRK', 'AVGO', 'COST', 'PEP', 'KO', 'WMT', 'ADBE',
    'MCD', 'CSCO', 'CRM', 'ACN', 'TMO', 'ABT', 'NFLX', 'DIS', 'VZ',
    'CMCSA', 'PFE', 'DHR', 'INTC', 'NEE', 'NKE', 'WFC', 'TXN', 'UPS',
    'PM', 'RTX', 'ORCL', 'BMY', 'QCOM', 'HON', 'UNP', 'LIN', 'MS',
    'AMD', 'T', 'SBUX', 'BA', 'AMGN', 'IBM', 'CAT', 'GE', 'LOW',
    'SPGI', 'ISRG', 'DE', 'BLK', 'MDT', 'AXP', 'GILD', 'TJX', 'MMC',
    'PLD', 'CVS', 'AMT', 'C', 'SYK', 'BKNG', 'ZTS', 'ADP', 'MDLZ',
    'CB', 'MO', 'VRTX', 'REGN', 'SO', 'ADI', 'CI', 'DUK', 'SCHW',
    'EOG', 'LRCX', 'BDX', 'PNC', 'NOC', 'TGT', 'USB', 'FI', 'BSX',
    'SLB', 'MU', 'EL', 'HUM', 'ITW', 'APD', 'EQIX', 'CL', 'AON'
);

-- Dow 30
UPDATE symbol
SET dow30 = true
WHERE ticker IN (
    'AAPL', 'MSFT', 'UNH', 'GS', 'HD', 'MCD', 'CAT', 'AMGN', 'V',
    'CRM', 'BA', 'TRV', 'AXP', 'HON', 'IBM', 'JPM', 'CVX', 'WMT',
    'JNJ', 'PG', 'DIS', 'MMM', 'NKE', 'KO', 'MRK', 'CSCO', 'DOW',
    'INTC', 'VZ', 'WBA'
);

-- NASDAQ-100 (sample)
UPDATE symbol
SET nasdaq100 = true
WHERE ticker IN (
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    'AVGO', 'COST', 'ADBE', 'NFLX', 'CSCO', 'PEP', 'AMD', 'INTC',
    'QCOM', 'CMCSA', 'TXN', 'AMGN', 'HON', 'SBUX', 'INTU', 'ISRG',
    'BKNG', 'GILD', 'ADP', 'VRTX', 'REGN', 'ADI', 'LRCX', 'MDLZ',
    'MU', 'PANW', 'SNPS', 'MELI', 'KLAC', 'CDNS', 'ASML', 'PYPL',
    'MAR', 'ORLY', 'MNST', 'CRWD', 'FTNT', 'AEP', 'CTAS', 'MRVL',
    'WDAY', 'NXPI', 'PAYX', 'ABNB', 'ROST', 'DXCM', 'ADSK', 'CPRT',
    'LULU', 'EXC', 'AZN', 'CHTR', 'MCHP', 'ON', 'KDP', 'FANG', 'TEAM',
    'CSGP', 'IDXX', 'GEHC', 'EA', 'MRNA', 'FAST', 'ODFL', 'PCAR',
    'BKR', 'XEL', 'WBD', 'BIIB', 'DDOG', 'ANSS', 'TTD', 'ILMN',
    'ZS', 'SIRI', 'WBA', 'ALGN', 'ENPH', 'JD', 'SGEN', 'LCID',
    'ZM', 'RIVN'
);

-- Show summary statistics
SELECT
    'Market Cap Tiers' as metric,
    market_cap_tier as tier,
    COUNT(*) as count
FROM symbol
WHERE market_cap_tier IS NOT NULL
GROUP BY market_cap_tier
ORDER BY
    CASE market_cap_tier
        WHEN 'mega' THEN 1
        WHEN 'large' THEN 2
        WHEN 'mid' THEN 3
        WHEN 'small' THEN 4
        WHEN 'micro' THEN 5
        WHEN 'nano' THEN 6
    END;

SELECT
    'Data Quality' as metric,
    CASE
        WHEN data_quality_score >= 80 THEN 'Excellent (80+)'
        WHEN data_quality_score >= 60 THEN 'Good (60-79)'
        WHEN data_quality_score >= 40 THEN 'Fair (40-59)'
        WHEN data_quality_score >= 20 THEN 'Poor (20-39)'
        ELSE 'Very Poor (<20)'
    END as quality_level,
    COUNT(*) as count
FROM symbol
WHERE data_quality_score IS NOT NULL
GROUP BY
    CASE
        WHEN data_quality_score >= 80 THEN 'Excellent (80+)'
        WHEN data_quality_score >= 60 THEN 'Good (60-79)'
        WHEN data_quality_score >= 40 THEN 'Fair (40-59)'
        WHEN data_quality_score >= 20 THEN 'Poor (20-39)'
        ELSE 'Very Poor (<20)'
    END
ORDER BY MIN(data_quality_score) DESC;

SELECT
    'Index Membership' as metric,
    SUM(CASE WHEN sp500 = true THEN 1 ELSE 0 END) as sp500_count,
    SUM(CASE WHEN dow30 = true THEN 1 ELSE 0 END) as dow30_count,
    SUM(CASE WHEN nasdaq100 = true THEN 1 ELSE 0 END) as nasdaq100_count
FROM symbol;
