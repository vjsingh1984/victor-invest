# TOON Integration Priority Ranking

Based on demonstrated token savings, here's the optimal rollout order:

## Priority 1: Technical Agent 🔥 HIGHEST IMPACT
**Savings: 29,200 tokens (87% reduction) per comprehensive analysis**

### Current Payload
- 90 days of historical data
- 60+ indicators (OHLCV, SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- JSON: ~33,750 tokens
- TOON: ~4,550 tokens

### Implementation
```python
# src/investigator/domain/agents/technical/agent.py
from src.investigator.domain.services.toon_formatter import to_toon_array

# Convert technical indicators to TOON
indicators_toon = to_toon_array(technical_data, name="price_data")

# Update prompt
prompt = f"""
{TOONFormatter.get_format_explanation()}

Analyze this technical data:
{indicators_toon}

Provide signals for: trend, support/resistance, momentum, volatility
"""
```

**Why First**: Single biggest token reduction, simplest tabular structure

---

## Priority 2: Fundamental Agent 🔥 HIGH IMPACT
**Savings: 2,000 tokens (63% reduction) per analysis**

### Current Payload
- 12 quarters of financial data
- 30+ metrics per quarter (revenue, margins, cash flows, ratios)
- JSON: ~3,200 tokens
- TOON: ~1,200 tokens

### Implementation
```python
# src/investigator/domain/agents/fundamental/agent.py
from src.investigator.domain.services.toon_formatter import to_toon_quarterly

# Convert quarterly data to TOON
quarterly_toon = to_toon_quarterly(quarters, metrics=[
    'fiscal_year', 'fiscal_period', 'period_end_date',
    'revenue', 'net_income', 'operating_cash_flow', 'free_cash_flow',
    'total_assets', 'total_liabilities', 'stockholders_equity',
    'earnings_per_share', 'shares_outstanding'
])
```

**Why Second**: High frequency (every analysis), proven 63% savings

---

## Priority 3: Synthesis Agent 🔥 HIGH IMPACT
**Savings: 1,500 tokens (55% reduction) per analysis**

### Current Payload
- Aggregated results from 4+ agents
- Peer comparison (10-15 companies)
- Historical performance
- JSON: ~2,700 tokens
- TOON: ~1,200 tokens

### Implementation
```python
# src/investigator/domain/agents/synthesis.py
from src.investigator.domain.services.toon_formatter import to_toon_peers, to_toon_array

# Peer comparison
peers_toon = to_toon_peers(peer_companies, metrics=[
    'symbol', 'market_cap', 'pe_ratio', 'ps_ratio', 
    'revenue_growth', 'profit_margin', 'fcf_yield'
])

# Agent summaries
agent_results_toon = to_toon_array([
    {"agent": "fundamental", "fair_value": 185.50, "confidence": 0.85},
    {"agent": "technical", "signal": "BUY", "strength": 0.75},
    {"agent": "market_context", "sector_rank": 3, "relative_strength": 1.15}
], name="agent_results")
```

**Why Third**: Coordinates all agents, benefits from multiple TOON tables

---

## Priority 4: Market Context Agent ⚡ MEDIUM IMPACT
**Savings: 600 tokens (45% reduction) per analysis**

### Current Payload
- Sector multiples (10-20 metrics)
- ETF holdings (top 10-20)
- Historical sector performance
- JSON: ~1,300 tokens
- TOON: ~700 tokens

### Implementation
```python
# src/investigator/domain/agents/market_context.py
from src.investigator.domain.services.toon_formatter import to_toon_array

# Sector multiples
sector_multiples_toon = to_toon_array(sector_data, name="sector_multiples")

# ETF holdings
etf_holdings_toon = to_toon_array(holdings, name="etf_holdings")
```

**Why Fourth**: Lower frequency, smaller datasets, but still solid savings

---

## Priority 5: Valuation Models ⚡ MEDIUM IMPACT
**Savings: 400 tokens (40% reduction) per analysis**

### Current Payload
- Historical cash flows (5-10 years)
- Dividend history
- Projection scenarios
- JSON: ~1,000 tokens
- TOON: ~600 tokens

### Implementation
```python
# utils/dcf_valuation.py
from src.investigator.domain.services.toon_formatter import to_toon_array

# Free cash flow history
fcf_history_toon = to_toon_array(fcf_data, name="fcf_history")

# utils/gordon_growth_model.py
dividend_history_toon = to_toon_array(dividend_data, name="dividend_history")
```

**Why Fifth**: Less frequent (only when applicable), smaller datasets

---

## Combined Impact

### Per Analysis Savings

**Standard Mode** (quick):
| Agent | Savings |
|-------|---------|
| Technical (30 days) | 850 tokens |
| Fundamental | 2,000 tokens |
| Synthesis | 800 tokens |
| **Total** | **3,650 tokens** |

**Comprehensive Mode**:
| Agent | Savings |
|-------|---------|
| Technical (90 days, 60 indicators) | 29,200 tokens |
| Fundamental | 2,000 tokens |
| Synthesis | 1,500 tokens |
| Market Context | 600 tokens |
| Valuation | 400 tokens |
| **Total** | **33,700 tokens** |

### Annual Cost Savings

At $0.01/1K tokens (local model equivalent):

| Volume | Mode | Savings/Analysis | Annual Savings |
|--------|------|------------------|----------------|
| 100/day | Standard | 3,650 tokens | **$1,332/year** |
| 100/day | Comprehensive | 33,700 tokens | **$12,300/year** |
| 1,000/day | Standard | 3,650 tokens | **$13,322/year** |
| 1,000/day | Comprehensive | 33,700 tokens | **$123,000/year** |

### Rollout Timeline

**Week 1**: Technical Agent
- Implement TOON formatter integration
- Test with 10 symbols across sectors
- Measure actual token savings
- Verify signal quality unchanged

**Week 2**: Fundamental Agent
- Integrate quarterly data TOON formatting
- Add TOON explanation to system prompt
- A/B test vs JSON baseline
- Deploy if quality matches

**Week 3**: Synthesis Agent
- Update peer comparison formatting
- Update agent aggregation
- Test cross-agent correlation quality
- Deploy if synthesis quality maintained

**Week 4**: Market Context + Valuation
- Implement remaining agents
- Full system testing
- Production deployment
- Monitor for 1 week

**Week 5**: Measurement & Optimization
- Measure actual token savings
- Calculate real cost impact
- Identify additional TOON opportunities
- Document lessons learned

---

## Success Metrics

1. **Token Reduction**: Achieve 50%+ reduction on tabular prompts
2. **Quality Maintenance**: Response quality within 5% of JSON baseline
3. **LLM Compatibility**: 100% parsing success across all models
4. **Cost Savings**: $1,000+/year verified savings
5. **Developer Experience**: TOON formatter easy to use, well-documented

---

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| LLM can't parse TOON | Test with multiple models first, add format explanation to system prompt |
| Quality degradation | A/B test thoroughly, rollback if quality drops >5% |
| Developer confusion | Clear docs, examples, helper functions |
| Cache invalidation | Version prompts, clear cache on TOON rollout |

---

## Rollback Plan

1. **Feature Flag**: `USE_TOON_FORMAT = True` in config
2. **Per-Agent Toggle**: Enable/disable TOON per agent
3. **Automatic Fallback**: If parsing error detected, retry with JSON
4. **Monitoring**: Track TOON parsing failures, alert if >1%

---

## Next Actions

- [x] Create TOON formatter utility
- [x] Validate token savings (demonstrated 42-87% reduction)
- [x] Document integration plan
- [ ] **START HERE**: Integrate technical agent (highest impact)
- [ ] Test technical agent with 10 symbols
- [ ] Measure real token savings
- [ ] Roll out to fundamental agent
- [ ] Complete remaining agents

