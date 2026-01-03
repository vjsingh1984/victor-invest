# Executive Summary Generator

## Overview

The ExecutiveSummaryGenerator reduces large analysis JSON files (typically 4.5MB) to concise executive summaries (~50KB) suitable for:
- **PDF Report Generation**: Provides key insights at the top of PDF reports
- **Database Persistence**: Stores summaries in `quarterly_ai_summaries` table for SQL-based opportunity screening across tickers

## Features

- **Deterministic Data Extraction**: Extracts key metrics from full analysis (market cap, P/E, ROE, valuation, technical indicators)
- **LLM-Powered Synthesis**: Uses DeepSeek-R1 32B model to generate narrative investment thesis, bull/bear cases, catalysts, and risks
- **Automatic Compression**: Reduces JSON size by ~99% (4.5MB → 50KB)
- **Database Integration**: Saves to existing `quarterly_ai_summaries` table with fiscal period tracking
- **Fallback Logic**: Rule-based summary generation if LLM fails

## Architecture

### Data Flow

```
Full Analysis JSON (4.5MB)
   ↓
ExecutiveSummaryGenerator
   ↓
Step 1: Extract Key Data (Deterministic)
   - Market cap, revenue, net income
   - P/E, ROE, debt/equity ratios
   - Current price, fair value
   - Technical trend, RSI
   - Quality scores, confidence levels
   ↓
Step 2: LLM Synthesis (deepseek-r1:32b)
   - Executive summary (3-4 sentences)
   - Investment recommendation (BUY/HOLD/SELL)
   - Conviction level (HIGH/MEDIUM/LOW)
   - Price target & upside potential
   - Bull case (3 points)
   - Bear case (3 points)
   - Top 3 catalysts
   - Top 3 risks
   ↓
Step 3: Combine & Save
   - Merge extracted data + LLM summary
   - Save to JSON file (~50KB)
   - Save to database (quarterly_ai_summaries)
```

### Database Schema

Executive summaries are stored in the `quarterly_ai_summaries` table:

```sql
CREATE TABLE quarterly_ai_summaries (
    id SERIAL PRIMARY KEY,
    cik VARCHAR(10) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_period VARCHAR(10) NOT NULL,  -- 'Q1', 'Q2', 'Q3', 'Q4', 'FY'
    form_type VARCHAR(10),                -- '10-Q' or '10-K'
    financial_summary TEXT,               -- Plain text summary
    ai_analysis JSONB,                    -- Full executive summary as JSON
    scores JSONB,                         -- Key metrics & scores
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (cik, fiscal_year, fiscal_period)
);
```

**Key Columns**:
- `ai_analysis`: Full executive summary JSON with recommendation, thesis, metrics
- `scores`: Extracted key metrics, valuation, financial health, technical summary
- `financial_summary`: Plain text summary for backward compatibility

## Usage

### 1. Generate Summary from Existing Analysis

```python
import asyncio
from utils.executive_summary_generator import generate_executive_summary_from_file
from core.ollama_client import OllamaClient

async def main():
    ollama = OllamaClient()

    # Database config
    db_config = {
        'host': '${DB_HOST:-localhost}',
        'user': 'investigator',
        'password': 'investigator',
        'database': 'sec_database'
    }

    # Generate summary with database persistence
    summary = await generate_executive_summary_from_file(
        input_file='results/NEE_FINAL_FIX.json',
        output_file='results/NEE_summary.json',
        ollama_client=ollama,
        db_config=db_config,  # Enable database persistence
        fiscal_year=2024,
        fiscal_period='Q3'
    )

    print(f"Summary size: {summary['summary_size_kb']}KB")
    print(f"Recommendation: {summary['recommendation']['action']}")

asyncio.run(main())
```

### 2. Using the Test Script

```bash
# Basic usage
python3 scripts/test_executive_summary.py results/NEE_FINAL_FIX.json

# Output:
# ✅ Executive Summary Generated Successfully!
#    Original size: 4.52MB
#    Summary size: 48.23KB
#    Compression: 1.0% of original
#
# 📄 Summary saved to: results/NEE_summary.json
# 💾 Summary saved to database: quarterly_ai_summaries table
#
# Key Metrics:
#   - Recommendation: BUY
#   - Conviction: MEDIUM
#   - Price Target: $85.50
#   - Current Price: $72.30
#   - Upside Potential: 18.3%
```

### 3. Querying Summaries from Database

```sql
-- Find all BUY recommendations with HIGH conviction
SELECT
    ticker,
    fiscal_year,
    fiscal_period,
    ai_analysis->>'recommendation'->>'action' AS recommendation,
    ai_analysis->>'recommendation'->>'conviction' AS conviction,
    ai_analysis->>'recommendation'->>'upside_potential' AS upside
FROM quarterly_ai_summaries
WHERE ai_analysis->>'recommendation'->>'action' = 'BUY'
  AND ai_analysis->>'recommendation'->>'conviction' = 'HIGH'
ORDER BY (ai_analysis->>'recommendation'->>'upside_potential')::float DESC
LIMIT 10;

-- Find stocks with strongest fundamentals
SELECT
    ticker,
    fiscal_year,
    fiscal_period,
    scores->'financial_health'->>'quality_score' AS quality_score,
    scores->'valuation'->>'discount_premium' AS discount_premium
FROM quarterly_ai_summaries
WHERE (scores->'financial_health'->>'quality_score')::float > 70
  AND (scores->'valuation'->>'discount_premium')::float > 15
ORDER BY (scores->'financial_health'->>'quality_score')::float DESC
LIMIT 10;

-- Get full executive summary for a ticker
SELECT
    ticker,
    fiscal_year,
    fiscal_period,
    ai_analysis
FROM quarterly_ai_summaries
WHERE ticker = 'NEE'
  AND fiscal_year = 2024
  AND fiscal_period = 'Q3';
```

## Output Structure

### Executive Summary JSON

```json
{
  "symbol": "NEE",
  "generated_at": "2025-11-03T02:15:00",
  "summary_version": "1.0",
  "original_size_mb": 4.52,
  "summary_size_kb": 48.23,

  "executive_summary": "NextEra Energy demonstrates strong fundamentals...",

  "recommendation": {
    "action": "BUY",
    "conviction": "MEDIUM",
    "price_target": 85.50,
    "current_price": 72.30,
    "upside_potential": 18.3,
    "time_horizon": "12 months"
  },

  "key_metrics": {
    "market_cap": 150000000000,
    "revenue": 20000000000,
    "net_income": 5000000000,
    "pe_ratio": 25.12,
    "roe": 12.34,
    "debt_to_equity": 1.45,
    "free_cash_flow": 3000000000
  },

  "investment_thesis": {
    "bull_case": [
      "Strong regulatory moat in utility sector",
      "Growing renewable energy portfolio",
      "Consistent dividend growth history"
    ],
    "bear_case": [
      "Interest rate sensitivity",
      "Regulatory risk",
      "High debt levels"
    ],
    "catalysts": [
      "Rate case approvals",
      "Renewable capacity expansion",
      "Federal tax credit extensions"
    ],
    "risks": [
      "Rising interest rates",
      "Weather-related disruptions",
      "Regulatory changes"
    ]
  },

  "valuation": {
    "fair_value": 85.50,
    "current_price": 72.30,
    "discount_premium": 18.3
  },

  "financial_health": {
    "quality_score": 75,
    "data_quality": "good",
    "confidence": 85
  },

  "technical_summary": {
    "trend": "bullish",
    "recommendation": "buy",
    "support": 68.50,
    "resistance": 78.00,
    "rsi": 58.5
  },

  "metadata": {
    "analysis_date": "2025-11-02T10:30:00",
    "analysis_mode": "standard",
    "agents_used": ["fundamental", "technical", "synthesis"],
    "data_quality": "good"
  }
}
```

## Integration with PDF Reports

The executive summary is designed to appear at the **TOP** of PDF reports:

1. **Executive Summary Section**: LLM-generated 3-4 sentence overview
2. **Investment Recommendation**: Action, conviction, price target, upside
3. **Investment Thesis Table**: Bull/bear cases, catalysts, risks
4. **Key Metrics Table**: Market cap, P/E, ROE, cash flow, etc.
5. **Valuation Summary**: Fair value vs. current price, discount/premium

## Performance

- **First Run (no cache)**: 10-15 seconds for LLM synthesis
- **Cached Run**: <1 second for deterministic extraction + cached LLM response
- **Compression Ratio**: ~99% (4.5MB → 50KB)
- **LLM Context**: ~2000 tokens input, ~1500 tokens output

## Error Handling

### LLM Failure Fallback

If LLM synthesis fails, the generator automatically falls back to rule-based summary:

```python
# Fallback logic
if quality_score > 70 and discount_premium > 15:
    recommendation = 'BUY'
    conviction = 'HIGH'
elif quality_score > 60 and discount_premium > 0:
    recommendation = 'BUY'
    conviction = 'MEDIUM'
elif quality_score < 40 or discount_premium < -20:
    recommendation = 'SELL'
    conviction = 'MEDIUM'
else:
    recommendation = 'HOLD'
    conviction = 'MEDIUM'
```

### Missing CIK

If CIK is not found for a symbol, database persistence is skipped but file-based summary is still generated:

```
⚠️  Cannot save summary to database: CIK not found for SYMBOL
✅ Summary still saved to file: results/SYMBOL_summary.json
```

## Development

### Adding New Metrics

To add new metrics to the executive summary:

1. **Update `_extract_key_data()` method**:
   ```python
   extracted['key_metrics']['new_metric'] = fundamental.get('new_metric', 0)
   ```

2. **Update LLM prompt** in `_synthesize_with_llm()`:
   ```python
   prompt += f"- New Metric: {extracted_data['key_metrics'].get('new_metric', 0):.2f}\n"
   ```

3. **Update database schema** if needed:
   ```sql
   -- Metrics are stored in JSONB, so schema changes are optional
   ```

### Testing

```bash
# Test with existing analysis file
python3 scripts/test_executive_summary.py results/NEE_FINAL_FIX.json

# Verify database persistence
PGPASSWORD=investigator psql -h ${DB_HOST:-localhost} -U investigator -d sec_database \
  -c "SELECT ticker, fiscal_year, fiscal_period, ai_analysis->>'recommendation' FROM quarterly_ai_summaries WHERE ticker = 'NEE';"
```

## Related Files

- `utils/executive_summary_generator.py`: Core generator class
- `scripts/test_executive_summary.py`: Test/demo script
- `dao/sec_bulk_dao.py`: Database access (CIK lookup)
- `agents/synthesis_agent.py`: Full analysis synthesis (input to this module)

## Future Enhancements

- [ ] Integrate with PDF generator to auto-include executive summary at top
- [ ] Add CLI command: `python3 cli_orchestrator.py summarize SYMBOL`
- [ ] Support batch summarization: `python3 cli_orchestrator.py summarize-batch AAPL MSFT GOOGL`
- [ ] Add SQL views for common opportunity screening queries
- [ ] Cache executive summaries separately from full analysis
- [ ] Add executive summary to email/Slack notifications
