# TOON Format Integration Plan

## Overview

TOON (Token-Oriented Object Notation) will be integrated for **input prompts only** to achieve 30-60% token savings on tabular data. Responses will remain in JSON format for parsing compatibility.

## Proven Savings

**Test Results** (4 quarters × 6 metrics):
- JSON: 186 tokens
- TOON: 69 tokens
- Savings: **117 tokens (63% reduction)**

**Projected Savings** (typical InvestiGator prompt):
- 12 quarters × 30 metrics
- Estimated savings: **~2,000 tokens per analysis**
- Scale: **200,000 tokens/day** (100 analyses)

## Integration Points

### 1. Fundamental Analysis Agent
**Location**: `src/investigator/domain/agents/fundamental/agent.py`

**Current JSON Payload**:
```python
quarterly_data = json.dumps(quarters, indent=2)  # ~3,500 tokens
prompt = f"Analyze this quarterly data:\n{quarterly_data}"
```

**TOON Conversion**:
```python
from src.investigator.domain.services.toon_formatter import to_toon_quarterly

# Convert to TOON
toon_data = to_toon_quarterly(quarters)  # ~1,500 tokens

# Add format explanation to system prompt
system_prompt = BASE_SYSTEM_PROMPT + TOONFormatter.get_format_explanation()

# User prompt uses TOON
prompt = f"Analyze this quarterly data:\n{toon_data}"
```

**Impact**: 57% token reduction on quarterly data prompts

### 2. Synthesis Agent
**Location**: `src/investigator/domain/agents/synthesis.py`

**Use Cases**:
- Peer comparison data
- Aggregated metrics across agents
- Market context data

**Implementation**:
```python
from src.investigator.domain.services.toon_formatter import to_toon_peers

# Peer data
toon_peers = to_toon_peers(peer_companies, metrics=[
    'symbol', 'market_cap', 'pe_ratio', 'ps_ratio', 
    'revenue_growth', 'profit_margin'
])
```

**Impact**: 45-55% token reduction on synthesis prompts

### 3. Market Context Agent  
**Location**: `src/investigator/domain/agents/market_context.py`

**Use Cases**:
- ETF holdings data
- Sector multiples
- Historical comparisons

**Implementation**:
```python
from src.investigator.domain.services.toon_formatter import to_toon_array

# Sector multiples
sector_multiples_toon = to_toon_array(sector_data, name="sector_multiples")
```

**Impact**: 40-50% token reduction on market context prompts

### 4. Valuation Prompts
**Location**: `utils/dcf_valuation.py`, `utils/gordon_growth_model.py`

**Use Cases**:
- Historical cash flows for DCF
- Dividend history for GGM

**Implementation**:
```python
# DCF historical data
fcf_history_toon = to_toon_array(fcf_data, name="free_cash_flow_history")

# GGM dividend history
dividend_history_toon = to_toon_array(dividend_data, name="dividend_history")
```

**Impact**: 35-45% token reduction on valuation prompts

## Implementation Phases

### Phase 1: Foundation (Complete ✅)
- [x] Create TOON formatter utility
- [x] Add format explanation generator
- [x] Validate token savings (63% confirmed)
- [x] Document integration points

### Phase 2: Core Agents (Next)
- [ ] Update fundamental analysis agent
- [ ] Add TOON explanation to system prompts
- [ ] Update synthesis agent for peer data
- [ ] Test with DASH/AAPL to verify LLM parsing

### Phase 3: Extended Coverage
- [ ] Update market context agent
- [ ] Update valuation utilities (DCF, GGM)
- [ ] Update technical analysis (if using tabular data)

### Phase 4: Validation & Rollout
- [ ] A/B test: TOON vs JSON on 10 symbols
- [ ] Verify response quality unchanged
- [ ] Measure actual token savings
- [ ] Enable for all analyses

## System Prompt Update

Add this to all agents using TOON:

```python
from src.investigator.domain.services.toon_formatter import TOONFormatter

# Append to existing system prompt
system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + TOONFormatter.get_format_explanation()
```

This explains the TOON format to the LLM so it can parse the data correctly.

## Response Format (Unchanged)

**CRITICAL**: Responses must remain JSON for parsing:

```python
response_instructions = """
Please respond in JSON format with this structure:
{
  "analysis": "...",
  "metrics": {...},
  "recommendation": "..."
}
"""
```

TOON is **input-only** - we still get structured JSON back for reliable parsing.

## Testing Strategy

### Unit Tests
```python
def test_toon_formatting():
    """Verify TOON formatter produces expected output"""
    data = [{"year": 2024, "revenue": 100}]
    toon = to_toon_quarterly(data)
    assert "quarterly_data[1]{year,revenue}:" in toon
    assert "2024,100" in toon

def test_token_savings():
    """Verify token savings vs JSON"""
    # ... token counting test
```

### Integration Tests
```python
def test_fundamental_agent_with_toon():
    """Verify agent can process TOON input and produce valid JSON output"""
    # Send TOON data
    result = agent.process(task_with_toon_data)
    # Verify JSON response
    assert isinstance(result.data, dict)
    assert "fair_value" in result.data
```

### LLM Parsing Validation
- Test with multiple LLMs (llama3.3:70b, qwen3:30b, etc.)
- Verify all can parse TOON correctly
- Compare analysis quality: TOON vs JSON

## Rollback Plan

If LLM parsing issues occur:
1. Feature flag: `USE_TOON_FORMAT = True/False` in config
2. Fallback to JSON if parsing errors detected
3. Log failures for debugging

## Expected Outcomes

### Token Savings
- **Fundamental analysis**: 2,000 tokens/analysis
- **Synthesis**: 800 tokens/analysis
- **Market context**: 600 tokens/analysis
- **Total**: ~3,400 tokens saved per complete analysis

### Cost Savings (at $0.01/1K tokens)
- Per analysis: $0.034 saved
- 100 analyses/day: $3.40/day = $102/month = **$1,224/year**
- 1,000 analyses/day: **$12,240/year**

### Performance
- Reduced prompt size = faster LLM processing
- Lower context window usage = more headroom for complex analyses

## Next Steps

1. Test fundamental agent with TOON on DASH symbol
2. Verify response quality matches JSON baseline
3. Measure actual token savings
4. Roll out to remaining agents
5. Update documentation with real-world results

## References

- TOON Formatter: `src/investigator/domain/services/toon_formatter.py`
- Test Results: 63% token savings demonstrated (186→69 tokens)
- Integration Target: All agents using tabular data in prompts
