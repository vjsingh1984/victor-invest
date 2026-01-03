# Free Data Sources for Investment Research

This document catalogs all free data sources used by Victor-Invest for investment research and analysis.

## Table of Contents

1. [Federal Reserve (FRED)](#federal-reserve-fred)
2. [SEC EDGAR](#sec-edgar)
3. [U.S. Treasury](#us-treasury)
4. [FINRA](#finra)
5. [Data Freshness](#data-freshness)

---

## Federal Reserve (FRED)

**Source:** [FRED API](https://fred.stlouisfed.org/docs/api/)
**API Key:** Required (free registration)
**Rate Limits:** 120 requests per minute

### Available Data Categories

| Category | Indicators | Update Frequency |
|----------|-----------|------------------|
| **Interest Rates** | DGS1MO, DGS3MO, DGS2, DGS10, DGS30, FEDFUNDS, DPRIME | Daily |
| **Credit Spreads** | BAMLC0A0CM, BAMLH0A0HYM2, BAA10Y, AAA10Y | Daily |
| **Volatility** | VIXCLS | Daily |
| **Economic** | GDP, GDPC1, UNRATE, CPIAUCSL, PCEPI, UMCSENT | Varies |
| **Markets** | SP500, DJIA, NASDAQCOM, WILL5000PR | Daily |
| **Money Supply** | M2SL, WALCL | Weekly |
| **Housing** | MORTGAGE30US, CSUSHPINSA | Weekly/Monthly |
| **Labor** | PAYEMS, ICSA, CCSA | Weekly/Monthly |

### Key Indicators

```bash
# Interest Rates
DGS10    - 10-Year Treasury Constant Maturity Rate
FEDFUNDS - Federal Funds Effective Rate
DPRIME   - Bank Prime Loan Rate

# Spreads & Credit
BAA10Y   - Moody's Baa Corporate Bond Yield Minus 10Y Treasury
AAA10Y   - Moody's Aaa Corporate Bond Yield Minus 10Y Treasury

# Volatility
VIXCLS   - CBOE Volatility Index (VIX)

# Economic Health
UNRATE   - Unemployment Rate
CPIAUCSL - Consumer Price Index
GDPC1    - Real Gross Domestic Product
```

### CLI Usage

```bash
./investigator_v2.sh --macro-summary                    # All macro indicators
./investigator_v2.sh --macro-category rates             # Interest rates only
./investigator_v2.sh --macro-indicators DGS10,VIX       # Specific indicators
./investigator_v2.sh --macro-time-series DGS10          # Historical data
```

---

## SEC EDGAR

**Source:** [SEC EDGAR](https://www.sec.gov/edgar)
**API Key:** Not required
**Rate Limits:** 10 requests per second

### Available Data

| Data Type | Filing Type | Update Frequency |
|-----------|-------------|------------------|
| Company Facts | XBRL | Quarterly |
| Insider Trading | Form 3/4/5 | Daily |
| Institutional Holdings | Form 13F | Quarterly |
| Financial Statements | 10-K, 10-Q | Quarterly |

### SEC Filing Types

```
Form 3   - Initial beneficial ownership statement
Form 4   - Changes in beneficial ownership
Form 5   - Annual changes in beneficial ownership
Form 13F - Quarterly institutional holdings (>$100M AUM)
10-K     - Annual report
10-Q     - Quarterly report
8-K      - Current report (material events)
```

### Data Endpoints

```
Company Facts:
https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json

Form 4 (Insider):
https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4

Form 13F:
https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F
```

### CLI Usage

```bash
# Insider Trading (Form 4)
./investigator_v2.sh --insider-sentiment AAPL           # Buy/sell sentiment
./investigator_v2.sh --insider-recent AAPL --days 90    # Recent transactions
./investigator_v2.sh --insider-clusters AAPL            # Coordinated activity

# Institutional Holdings (Form 13F)
./investigator_v2.sh --inst-holdings AAPL               # Ownership summary
./investigator_v2.sh --inst-top-holders AAPL            # Top institutions
./investigator_v2.sh --inst-institution 0001067983      # Berkshire holdings
```

---

## U.S. Treasury

**Source:** [Treasury Fiscal Data API](https://fiscaldata.treasury.gov/api-documentation/)
**API Key:** Not required
**Rate Limits:** None specified

### Available Data

| Data Type | Maturities | Update Frequency |
|-----------|-----------|------------------|
| Yield Curve | 1M to 30Y | Daily |
| Historical Yields | All maturities | Daily |

### Yield Curve Maturities

```
1-Month   (1m)     2-Year  (2y)      10-Year (10y)
3-Month   (3m)     5-Year  (5y)      20-Year (20y)
6-Month   (6m)     7-Year  (7y)      30-Year (30y)
1-Year    (1y)
```

### Key Spreads

```
10Y-2Y Spread  - Classic recession indicator
10Y-3M Spread  - NY Fed recession probability input
```

### CLI Usage

```bash
./investigator_v2.sh --treasury-curve                   # Current yield curve
./investigator_v2.sh --treasury-spread                  # Spread analysis
./investigator_v2.sh --treasury-recession               # Recession probability
./investigator_v2.sh --treasury-history --maturity 10y  # Historical 10Y yields
```

---

## FINRA

**Source:** [FINRA Data API](https://api.finra.org/data/)
**API Key:** Required (free registration)
**Rate Limits:** Varies by endpoint

### Available Data

| Data Type | Detail Level | Update Frequency |
|-----------|-------------|------------------|
| Short Interest | Symbol-level | Bi-monthly (1st & 15th) |
| Daily Short Volume | Symbol-level | Daily |
| Market-wide Stats | Aggregate | Daily |

### Short Interest Metrics

```
Short Interest      - Total shares sold short
Days to Cover       - Short Interest / Avg Daily Volume
Short % of Float    - Short Interest / Float
Short % of Shares   - Short Interest / Shares Outstanding
```

### Squeeze Indicators

```
Days to Cover > 5       - Potential squeeze setup
Short % of Float > 20%  - High short interest
Rising Short Interest   - Building bearish sentiment
```

### CLI Usage

```bash
./investigator_v2.sh --short-current GME                # Current short interest
./investigator_v2.sh --short-history GME                # Historical data
./investigator_v2.sh --short-squeeze GME                # Squeeze risk analysis
./investigator_v2.sh --short-most-shorted               # Most shorted stocks
```

---

## Data Freshness

### Update Schedule

| Data Source | Update Time (ET) | CLI Flag |
|-------------|-----------------|----------|
| FRED Macro | 9:00 AM | `--macro-*` |
| Treasury Yields | 6:00 PM | `--treasury-*` |
| SEC Form 4 | Every 4 hours | `--insider-*` |
| SEC Form 13F | 7:00 AM | `--inst-*` |
| FINRA Short Interest | 10:00 AM (1st/15th) | `--short-*` |

### Cache Behavior

```bash
# Force fresh data (bypass cache)
./investigator_v2.sh --macro-summary --force-refresh

# Check data age
./investigator_v2.sh --inspect-cache
```

### Data Latency

| Data Type | Typical Delay |
|-----------|--------------|
| Treasury Yields | Same day |
| FRED Indicators | 1-2 days |
| Insider Trades (Form 4) | 2 business days |
| 13F Holdings | 45 days after quarter |
| Short Interest | 11-14 days |

---

## Integration Notes

### API Keys Required

```bash
# Set in environment or config.yaml
FRED_API_KEY=your_key_here
FINRA_API_KEY=your_key_here  # Optional for some endpoints
```

### Rate Limiting

The system automatically handles rate limiting with exponential backoff:

```python
# Default settings in config.yaml
rate_limits:
  fred:
    requests_per_minute: 120
  sec:
    requests_per_second: 10
  treasury:
    requests_per_second: 10
```

### Error Handling

All data fetchers include:
- Automatic retry with exponential backoff
- Graceful degradation when data unavailable
- Cache fallback for stale data
- Detailed error logging

---

## See Also

- [Credit Risk Models](./CREDIT_RISK_MODELS.md)
- [Data Collection Schedule](./DATA_COLLECTION_SCHEDULE.md)
- [CLI Data Commands](./CLI_DATA_COMMANDS.md)
