# Architecture Decision Record: Data Access Strategy

## Decision: Hybrid Direct-DB + LLM Tool Calling

**Status**: Accepted
**Date**: 2025-12-29
**Context**: Victor-Invest vertical migration to victor-core framework

---

## Executive Summary

Victor-Invest uses a **hybrid data access strategy**:

1. **Direct Database Access** for deterministic data collection phases
2. **LLM Tool Calling** for on-demand queries during analysis/reasoning phases

This decision optimizes for accuracy, cost, and latency in financial analysis workflows.

---

## Context

Victor-Core provides two mechanisms for tools to access data:

### Option A: Direct Database Access (Tools → DB)
```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│ StateGraph  │────▶│ SECFilingTool│────▶│ PostgreSQL │
│   Node      │     │ (direct SQL) │     │  Database  │
└─────────────┘     └──────────────┘     └────────────┘
```

### Option B: LLM Tool Calling (Tools → LLM → DB)
```
┌─────────────┐     ┌─────────┐     ┌──────────────┐     ┌────────────┐
│ Agent with  │────▶│   LLM   │────▶│ DatabaseTool │────▶│ PostgreSQL │
│   Prompt    │     │ (Ollama)│     │ (via invoke) │     │  Database  │
└─────────────┘     └─────────┘     └──────────────┘     └────────────┘
```

---

## Decision Rationale

### Why Direct Database Access for Data Collection?

Financial analysis requires **deterministic data retrieval**. For the data collection phase:

| Criterion | Direct DB | LLM Tool Calling |
|-----------|-----------|------------------|
| **Accuracy** | ✅ Deterministic queries | ⚠️ LLM may hallucinate params |
| **Latency** | ✅ ~50-200ms per query | ❌ +2-10s LLM inference |
| **Cost** | ✅ $0 (local DB only) | ❌ Token usage per query |
| **Predictability** | ✅ Same query = same result | ⚠️ Temperature affects output |
| **Offline** | ✅ Works without LLM | ❌ Requires LLM server |
| **Auditability** | ✅ SQL queries logged | ⚠️ LLM reasoning opaque |

**For investment analysis where incorrect data = incorrect recommendations:**
- Revenue/earnings figures must be exact
- Fiscal period alignment must be deterministic
- Cache key generation must be consistent

### When LLM Tool Calling Makes Sense

Tool calling is valuable for **on-demand, context-dependent queries**:

1. **Peer Comparison**: "Get 5 similar companies to AAPL in the same sector"
2. **Dynamic Exploration**: "Find the most impactful metric for this company"
3. **Clarification**: "User asked about 'growth' - is this revenue or earnings growth?"
4. **Fallback Recovery**: "Primary data source failed, try alternative approach"

These scenarios require LLM judgment to determine *what* to query.

---

## Implementation Architecture

### Phase 1: Deterministic Data Collection (Direct DB)

```python
# victor_invest/workflows/graphs.py

async def fetch_sec_data(state_input) -> dict:
    """
    ARCHITECTURE DECISION: Direct Database Access

    Why not LLM tool calling here?
    - SEC data retrieval is deterministic (given symbol → get facts)
    - No reasoning needed to decide WHAT to fetch
    - Latency critical (parallel data fetching)
    - Accuracy critical (financial figures must be exact)

    The workflow graph orchestrates WHEN to fetch, tools handle HOW.
    LLM reasoning happens in synthesis phase, not data collection.
    """
    state = _ensure_state(state_input)
    sec_tool = await _get_sec_tool()

    # Direct execution - no LLM round-trip
    facts_result = await sec_tool.execute(
        symbol=state.symbol,
        action="get_company_facts"
    )
    # ...
```

### Phase 2: LLM-Driven Analysis (Tool Calling Available)

```python
# Future: When LLM needs on-demand data during synthesis

async def run_synthesis_with_agent(state_input) -> dict:
    """
    ARCHITECTURE DECISION: LLM Tool Calling for Dynamic Queries

    Why tool calling here?
    - LLM may need additional context based on initial analysis
    - Peer comparison requires judgment (which peers are relevant?)
    - Follow-up queries depend on reasoning output

    Example: If valuation looks anomalous, LLM might call:
    - get_peer_valuations(sector="Technology", metric="P/E")
    - get_historical_range(symbol="AAPL", metric="P/E", years=5)
    """
    agent = await Agent.create(
        provider="ollama",
        model="qwen3:30b",
        tools=[SECFilingTool, ValuationTool, MarketDataTool],
    )
    # LLM can invoke tools as needed during reasoning
    result = await agent.run(synthesis_prompt)
```

---

## Tool Design for Dual-Mode Operation

Tools are designed to work in both modes:

```python
class SECFilingTool(BaseTool):
    """
    SEC Filing Tool - Dual-Mode Data Access

    MODE 1: Direct Invocation (StateGraph nodes)
        - Called by workflow graph during data collection
        - Deterministic: symbol → SEC data
        - No LLM involvement

        Example:
            tool = SECFilingTool()
            result = await tool.execute(symbol="AAPL", action="get_company_facts")

    MODE 2: LLM Tool Calling (Agent reasoning)
        - Registered with Victor Agent via ToolSet
        - LLM decides when to invoke based on context
        - Useful for dynamic/exploratory queries

        Example (in agent prompt):
            "If the P/E ratio seems anomalous, use sec_filing tool
             to get historical earnings data for comparison"

    DESIGN PRINCIPLE:
        Tools are stateless and mode-agnostic. The CALLER decides
        whether to invoke directly (deterministic) or via LLM (dynamic).
    """
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INVESTMENT ANALYSIS WORKFLOW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ PHASE 1: DATA COLLECTION (Direct DB - Deterministic)                │    │
│  │                                                                      │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │    │
│  │   │ fetch_sec    │    │ fetch_market │    │ (parallel)   │          │    │
│  │   │ _data()      │    │ _data()      │    │              │          │    │
│  │   └──────┬───────┘    └──────┬───────┘    └──────────────┘          │    │
│  │          │                   │                                       │    │
│  │          ▼                   ▼                                       │    │
│  │   ┌──────────────┐    ┌──────────────┐                              │    │
│  │   │ SECFilingTool│    │MarketDataTool│  ◀── Direct DB queries       │    │
│  │   │ .execute()   │    │ .execute()   │      No LLM involvement      │    │
│  │   └──────┬───────┘    └──────┬───────┘                              │    │
│  │          │                   │                                       │    │
│  │          ▼                   ▼                                       │    │
│  │   ┌──────────────────────────────────────┐                          │    │
│  │   │         PostgreSQL Databases          │                          │    │
│  │   │  sec_database  │  market_data_db     │                          │    │
│  │   └──────────────────────────────────────┘                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ PHASE 2: ANALYSIS (Direct Tool Execution)                           │    │
│  │                                                                      │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │    │
│  │   │ fundamental  │    │ technical    │    │ market       │          │    │
│  │   │ _analysis()  │    │ _analysis()  │    │ _context()   │          │    │
│  │   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │    │
│  │          │                   │                   │                   │    │
│  │          ▼                   ▼                   ▼                   │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │    │
│  │   │ValuationTool │    │TechnicalTool │    │MarketDataTool│          │    │
│  │   │(compute DCF) │    │(calc RSI,MA) │    │(get context) │          │    │
│  │   └──────────────┘    └──────────────┘    └──────────────┘          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ PHASE 3: SYNTHESIS (Weighted Scoring - No LLM Currently)            │    │
│  │                                                                      │    │
│  │   Current: Deterministic weighted scoring using SYNTHESIS_AGENT_SPEC│    │
│  │   Future:  LLM synthesis with tool calling for on-demand queries    │    │
│  │                                                                      │    │
│  │   ┌──────────────────────────────────────────────────────────────┐  │    │
│  │   │ run_synthesis()                                               │  │    │
│  │   │   - Combines fundamental + technical + market scores         │  │    │
│  │   │   - Applies SYNTHESIS_AGENT_SPEC weight distribution         │  │    │
│  │   │   - Produces BUY/HOLD/SELL recommendation                    │  │    │
│  │   └──────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ FUTURE: LLM-ENHANCED SYNTHESIS (Tool Calling for Dynamic Queries)  │    │
│  │                                                                      │    │
│  │   ┌────────────┐      ┌─────────────────────────────────────┐       │    │
│  │   │   Ollama   │◀────▶│ Tools available for on-demand calls │       │    │
│  │   │ qwen3:30b  │      │  - SECFilingTool (historical data)  │       │    │
│  │   │            │      │  - ValuationTool (peer comparison)  │       │    │
│  │   │ "Analyze   │      │  - MarketDataTool (sector context)  │       │    │
│  │   │  results   │      └─────────────────────────────────────┘       │    │
│  │   │  and..."   │                                                    │    │
│  │   └────────────┘                                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Database Credential Security

### Current Implementation

```yaml
# config.yaml - Uses environment variable substitution
database:
  password: ${DB_PASSWORD:-investigator}  # Fallback for dev only
```

```bash
# .env (gitignored) - Production credentials
DB_PASSWORD=actual_secure_password
```

### Security Layers

1. **Environment Variables**: Primary mechanism (`DB_PASSWORD`)
2. **Gitignored Files**: `.env`, `config.local.json`, `config.prod.json`
3. **Default Fallbacks**: Only for development, never production

### Future Enhancement Options

For production deployments, consider:

```python
# Option 1: Python keyring (OS credential store)
import keyring
password = keyring.get_password("victor-invest", "db_password")

# Option 2: AWS Secrets Manager
import boto3
client = boto3.client('secretsmanager')
secret = client.get_secret_value(SecretId='victor-invest/db')

# Option 3: HashiCorp Vault
import hvac
client = hvac.Client(url='https://vault.example.com')
secret = client.secrets.kv.read_secret_version(path='victor-invest/db')
```

---

## Trade-offs and Considerations

### Advantages of Current Hybrid Approach

1. **Accuracy**: Financial data is fetched deterministically
2. **Performance**: No LLM latency for data collection (~50ms vs ~5s)
3. **Cost**: No token usage for routine data fetching
4. **Auditability**: SQL queries are logged and reproducible
5. **Offline**: Data collection works without LLM server

### When to Consider More LLM Tool Calling

1. **Complex Queries**: "Find companies with similar growth profile to AAPL"
2. **Adaptive Analysis**: Different analysis based on company archetype
3. **User Interaction**: Conversational interface for ad-hoc queries
4. **Error Recovery**: LLM can try alternative data sources on failure

### Not Recommended

❌ Using LLM tool calling for:
- Basic symbol lookups
- Standard financial metric retrieval
- Fixed workflow data collection
- High-frequency/batch operations

---

---

## LLM Prompt Strategy: Context Stuffing vs Tool Calling

### The Fundamental Trade-off

When LLM needs data for analysis, two strategies exist:

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Context Stuffing** | Pre-fetch all data, include in prompt | Bounded, predictable analysis |
| **Tool Calling** | LLM fetches on-demand via tools | Exploratory, adaptive analysis |

---

### Strategy 1: Context Stuffing (Pre-fetch → Prompt)

```
┌────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Data Collection│────▶│ Build Prompt    │────▶│ LLM Inference   │
│ (Direct DB)    │     │ with ALL data   │     │ (Single call)   │
└────────────────┘     └─────────────────┘     └─────────────────┘
```

**Implementation in Victor-Invest:**

```python
# victor_invest/workflows/graphs.py

async def run_synthesis(state_input) -> dict:
    """
    ARCHITECTURE DECISION: Context Stuffing for Synthesis

    Why include all data in prompt (not tool calling)?

    1. BOUNDED SCOPE: We know exactly what data is needed
       - Fundamental analysis results (valuation models)
       - Technical analysis results (indicators)
       - Market context (sector performance)

    2. SINGLE INFERENCE: One LLM call with complete context
       - No back-and-forth tool calling latency
       - Predictable token usage
       - Easier to cache/reproduce

    3. STRUCTURED OUTPUT: We want specific format
       - BUY/HOLD/SELL recommendation
       - Confidence score
       - Supporting rationale

    4. NO EXPLORATION NEEDED: Analysis scope is fixed
       - User asked for analysis of AAPL
       - We don't need to discover what to analyze

    When to use tool calling instead:
    - "Compare AAPL to its peers" (which peers? LLM decides)
    - "Find anomalies in this data" (what to look for? LLM decides)
    - Interactive Q&A where follow-ups are unknown
    """
    state = _ensure_state(state_input)

    # Pre-collected data (from earlier workflow nodes)
    fundamental = state.fundamental_analysis
    technical = state.technical_analysis
    market = state.market_context

    # ALL data included in prompt - no tool calling needed
    synthesis_prompt = f'''
    Analyze {state.symbol} investment opportunity.

    FUNDAMENTAL ANALYSIS:
    {json.dumps(fundamental, indent=2)}

    TECHNICAL ANALYSIS:
    {json.dumps(technical, indent=2)}

    MARKET CONTEXT:
    {json.dumps(market, indent=2)}

    Provide: recommendation, confidence, rationale.
    '''

    # Single LLM call with complete context
    result = await llm.generate(synthesis_prompt)
```

**When to Use Context Stuffing:**

| Scenario | Rationale |
|----------|-----------|
| Standard stock analysis | Fixed data requirements, bounded scope |
| Report generation | All data known upfront, structured output |
| Batch processing | Predictable, parallelizable |
| Valuation synthesis | Combining known inputs into recommendation |
| Risk assessment | Fixed set of risk factors to evaluate |

---

### Strategy 2: Tool Calling (LLM Fetches On-Demand)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Initial Prompt  │────▶│ LLM Reasoning   │────▶│ Tool Invocation │
│ (minimal data)  │     │ "I need more..."│     │ (fetch data)    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │ LLM Continues   │◀─────────────┘
                        │ with new data   │
                        └─────────────────┘
```

**When Tool Calling is Appropriate:**

```python
# Example: Exploratory peer analysis (FUTURE ENHANCEMENT)

async def exploratory_analysis(symbol: str, user_question: str):
    """
    ARCHITECTURE DECISION: Tool Calling for Exploratory Analysis

    Why use tool calling here (not context stuffing)?

    1. UNBOUNDED SCOPE: We don't know what data is needed
       - User: "Why is AAPL underperforming?"
       - Could need: peer comparison, sector data, macro indicators

    2. ADAPTIVE REASONING: LLM determines analysis path
       - If P/E is high → fetch peer P/E ratios
       - If growth is low → fetch historical growth
       - If sector is weak → fetch sector ETF data

    3. TOKEN EFFICIENCY: Only fetch what's needed
       - Don't pre-load 100 metrics when LLM needs 5
       - Especially important for large context windows

    4. INTERACTIVE: Multi-turn exploration
       - User follow-ups require additional data
       - Can't pre-fetch for unknown questions
    """
    agent = await Agent.create(
        provider="ollama",
        model="qwen3:30b",
        tools=[
            SECFilingTool,      # For on-demand SEC data
            ValuationTool,      # For peer comparisons
            MarketDataTool,     # For sector context
            TechnicalTool,      # For historical patterns
        ],
    )

    # Minimal initial prompt - LLM will fetch as needed
    response = await agent.run(f'''
        User question about {symbol}: {user_question}

        You have tools to fetch:
        - SEC filings and financial metrics
        - Valuation comparisons
        - Market data and sector context
        - Technical indicators

        Investigate and answer the question.
        Use tools as needed to gather evidence.
    ''')
```

**When to Use Tool Calling:**

| Scenario | Rationale |
|----------|-----------|
| Interactive Q&A | Unknown follow-ups, adaptive exploration |
| Peer discovery | LLM decides which peers are relevant |
| Anomaly investigation | LLM determines what to investigate |
| Research tasks | Open-ended, exploratory analysis |
| Error recovery | LLM can try alternative data sources |

---

### Decision Matrix: Context Stuffing vs Tool Calling

```
                    BOUNDED SCOPE                    UNBOUNDED SCOPE
                    (Known data needs)               (Discovery needed)
                    │                                │
    SINGLE          │  ✅ CONTEXT STUFFING           │  ⚠️ HYBRID
    INFERENCE       │  - Standard analysis           │  - Initial context + tools
                    │  - Report generation           │  - Known base, unknown extras
                    │  - Batch processing            │
                    │                                │
    ─────────────────────────────────────────────────────────────────────
                    │                                │
    MULTI-TURN      │  ⚠️ HYBRID                     │  ✅ TOOL CALLING
    EXPLORATION     │  - Pre-fetch common data       │  - Interactive Q&A
                    │  - Tools for follow-ups        │  - Research exploration
                    │                                │  - Anomaly investigation
```

---

### Victor-Invest Prompt Patterns

#### Pattern 1: Analysis Prompt (Context Stuffing)

```python
# Used in: run_synthesis(), run_fundamental_analysis()

SYNTHESIS_PROMPT = '''
# Investment Analysis: {symbol}

## Data Provided (Complete - No Tool Calls Needed)

### Financial Metrics
{fundamental_data}

### Technical Indicators
{technical_data}

### Market Context
{market_data}

## Task
Synthesize the above data into an investment recommendation.

## Output Format
- Recommendation: BUY/HOLD/SELL
- Confidence: HIGH/MEDIUM/LOW
- Key Factors: [list 3-5 factors]
- Risk Assessment: [key risks]

## Constraints
- Base analysis ONLY on provided data
- Do not make assumptions about unavailable data
- Clearly state if data is insufficient
'''
```

#### Pattern 2: Exploratory Prompt (Tool Calling)

```python
# Used in: interactive_analysis(), research_mode()

EXPLORATION_PROMPT = '''
# Investment Research: {symbol}

## User Question
{user_question}

## Available Tools
You have access to these tools for on-demand data fetching:

1. sec_filing - Get SEC filings, company facts, financial metrics
2. valuation - Calculate fair values, compare to peers
3. technical_indicators - Get RSI, MACD, support/resistance
4. market_data - Get quotes, history, sector performance

## Instructions
1. Analyze the question to determine what data you need
2. Use tools to fetch relevant data
3. Synthesize findings into a clear answer
4. Cite specific data points in your response

## Constraints
- Only fetch data relevant to the question
- Limit tool calls to essential queries
- Acknowledge when data is unavailable
'''
```

#### Pattern 3: Hybrid Prompt (Base Context + Optional Tools)

```python
# Used in: enhanced_synthesis() (future)

HYBRID_PROMPT = '''
# Investment Analysis: {symbol}

## Pre-Loaded Context (Already Fetched)
{base_analysis_results}

## Available Tools (For Additional Context)
If your analysis reveals anomalies or requires comparison:
- valuation: Get peer valuations for comparison
- sec_filing: Get historical data if trend analysis needed
- market_data: Get sector/market context if relevant

## Task
1. Analyze the pre-loaded data
2. If something seems anomalous, use tools to investigate
3. Provide recommendation with supporting evidence

## Output Format
{output_schema}
'''
```

---

### Token Budget Considerations

| Approach | Typical Token Usage | When Optimal |
|----------|---------------------|--------------|
| Context Stuffing | 2K-8K input + 1K output | Bounded analysis, predictable data |
| Tool Calling | Variable (500-5K per round) | Exploration, may need multiple rounds |
| Hybrid | 2K base + variable tool calls | Best of both for complex analysis |

**Victor-Invest Defaults:**

```python
# config.yaml

llm:
  synthesis:
    strategy: "context_stuffing"  # All data in prompt
    max_input_tokens: 8000
    max_output_tokens: 2000

  exploration:
    strategy: "tool_calling"  # On-demand fetching
    max_tool_calls: 10
    max_rounds: 5

  research:
    strategy: "hybrid"  # Base context + tools
    base_context_tokens: 4000
    max_tool_calls: 15
```

---

### Implementation in Victor-Invest Tools

All tools are designed for dual-mode operation:

```python
class SECFilingTool(BaseTool):
    """
    SEC Filing Tool - Supports Both Context Stuffing and Tool Calling

    CONTEXT STUFFING MODE (workflow nodes):
        Called directly by StateGraph nodes during data collection.
        Results are included in synthesis prompt.

        # In fetch_sec_data node:
        result = await sec_tool.execute(symbol="AAPL", action="get_company_facts")
        state.sec_data = result.data  # Added to synthesis prompt

    TOOL CALLING MODE (agent exploration):
        Registered with Victor Agent for on-demand invocation.
        LLM decides when to call based on reasoning.

        # Agent decides to fetch more data:
        agent.tools = [SECFilingTool]
        # LLM: "I need historical earnings to verify this trend..."
        # → Invokes sec_filing tool with action="get_historical_metrics"

    DESIGN PRINCIPLE:
        Tools are mode-agnostic. The ORCHESTRATION LAYER decides:
        - StateGraph: Direct execution → Context stuffing
        - Agent: LLM-driven execution → Tool calling
    """
```

---

## Conclusion

The hybrid approach aligns with investment analysis requirements:

- **Deterministic data** → Direct database access
- **Bounded analysis** → Context stuffing (all data in prompt)
- **Exploratory analysis** → Tool calling (LLM fetches on-demand)
- **Security** → Environment variables, gitignored configs

This architecture provides institutional-grade reliability while preserving flexibility for future LLM-enhanced features.
