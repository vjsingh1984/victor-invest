# InvestiGator Codebase Architecture - Comprehensive Guide

**Version**: 2.0 (Clean Architecture Migration Complete)
**Last Updated**: 2025-11-04

> **🎯 Clean Architecture Migration**: As of November 2025, InvestiGator has migrated to Clean Architecture (Hexagonal Architecture) with clear separation between domain, infrastructure, application, and interface layers.
>
> **New Structure**: `src/investigator/{domain,infrastructure,application,interfaces,config}/`
> **Main CLI**: `cli_orchestrator.py` (root level)
> **Entry Points**:
>   - `investigator` command (via pyproject.toml)
>   - `python -m investigator` (via `__main__.py`)
>   - `./cli_orchestrator.py` (direct execution)
> **Tests**: `tests/unit/{domain,infrastructure,application,config}/`

## Executive Summary

InvestiGator is a sophisticated **AI-powered investment analysis system** that orchestrates multiple specialized agents working in concert to provide comprehensive financial analysis. The architecture is built on clean design patterns, efficient resource management, and a multi-layer caching system optimized for both performance and reliability.

**Key Characteristics:**
- **Agent-based architecture** with specialized agents (SEC, Technical, Fundamental, Synthesis)
- **Multi-layer caching** (File/Parquet + RDBMS) with priority-based retrieval
- **VRAM-aware resource management** for GPU-constrained LLM operations
- **Event-driven inter-agent communication** via event bus
- **Async-first design** for high concurrency and responsiveness

---

## 1. Agent System Architecture

### 1.1 Core Agent Framework (`agents/base.py`)

The foundation of the system is the **`InvestmentAgent`** abstract base class that all specialized agents inherit from.

**Key Components:**

```
InvestmentAgent (ABC)
├── agent_id: Unique agent identifier
├── ollama_client: LLM interface
├── event_bus: Inter-agent communication
├── cache_manager: Multi-layer cache access
├── metrics: Performance tracking (AgentMetrics)
└── capabilities: List[AgentCapability] - what agent can do
```

**Lifecycle Pattern** (`agents/base.py:199-205`):
```
create task → hydrate context → process() → emit AgentResult
    ↑              │                │
    └─────────────────── cache lookups/upserts
metrics/event bus captured externally
```

**Key Methods:**
- `async run(task)`: Main entry point - orchestrates entire task execution
- `async process(task)`: Abstract method implemented by concrete agents
- `can_handle_task(task)`: Check if agent has capabilities for task
- `pre_process(task)`: Validation, duplicate prevention
- `post_process(task, result)`: Cleanup, metrics update
- `execute_with_timeout()`: Timeout handling with fallback
- `execute_with_retry()`: Exponential backoff retry logic
- `_cache_llm_response()`: Separate caching for audit/debugging

**Agent Task Structure** (`agents/base.py:96-119`):
```python
@dataclass
class AgentTask:
    task_id: str
    symbol: str
    analysis_type: AnalysisType  # SEC_FUNDAMENTAL, TECHNICAL_ANALYSIS, etc.
    priority: Priority  # CRITICAL → LOW
    context: Dict[str, Any]  # Input data for agent
    timeout: Optional[int]  # Task timeout
    dependencies: List[str]  # Dependent task IDs
    status: TaskStatus  # PENDING → COMPLETED/FAILED
    
    def get_cache_key(self) -> str:
        """SHA256 hash of symbol + analysis_type + context keys"""
```

**Agent Result Structure** (`agents/base.py:122-151`):
```python
@dataclass
class AgentResult:
    task_id: str
    agent_id: str
    status: TaskStatus
    result_data: Dict[str, Any]  # Analysis output
    processing_time: float  # Seconds
    error: Optional[str]
    metadata: Dict[str, Any]
    cached: bool  # Was result cached?
    cache_hit: bool  # Did we hit cache?
    timestamp: datetime
```

### 1.2 Specialized Agent Implementations

#### **FundamentalAnalysisAgent** (`agents/fundamental_agent.py`)
Performs financial health analysis using SEC data.

**Capabilities:**
- Extracts quarterly financial metrics (revenue, net income, assets, etc.)
- Calculates valuation ratios (P/E, P/B, P/S, debt-to-equity, etc.)
- Analyzes financial health (ROE, ROA, margins, liquidity ratios)
- Generates LLM-powered fundamental analysis
- Tracks multi-quarter trends

**Data Structure - QuarterlyData** (`agents/fundamental_agent.py:23-68`):
```python
@dataclass
class QuarterlyData:
    fiscal_year: int  # 2024
    fiscal_period: str  # Q1, Q2, Q3, Q4, FY
    financial_data: Dict[str, Any]  # Core metrics
    ratios: Optional[Dict[str, Any]]  # Calculated ratios
    market_data: Optional[Dict[str, Any]]  # Price, market cap
    data_quality: Optional[Dict[str, Any]]  # Data completeness
    filing_date: Optional[str]
    
    @property
    def period_label(self) -> str:
        return f"{fiscal_year}-{fiscal_period}"  # E.g., "2024-Q1"
```

**Key Methods:**
- `async process(task)`: Main analysis pipeline
- `_extract_company_data()`: Fetch SEC financial data
- `_generate_fundamental_analysis()`: LLM-powered analysis
- `_calculate_ratios()`: Financial metric calculations
- `_assess_valuation()`: Valuation scoring

#### **TechnicalAnalysisAgent** (`agents/technical_agent.py`)
Analyzes price action and technical patterns.

**Pipeline**:
```
fetch OHLCV → compute indicators → prompt LLM → cache outputs
    ↑              │                    │
    └────────────────────────────── stores parquet snapshots
cache reuses DataFrames for future runs
```

**Key Methods:**
- `async process(task)`: Technical analysis pipeline
- `_fetch_price_data()`: Historical price/volume data
- `_compute_technical_indicators()`: RSI, MACD, Bollinger Bands, etc.
- `_detect_patterns()`: Head-and-shoulders, triangles, flags, etc.
- `_generate_technical_analysis()`: LLM interpretation

**Technical Timeframes**:
- Intraday (1d)
- Short-term (1mo)
- Medium-term (3mo)
- Long-term (1y)
- Extended (5y)

#### **SynthesisAgent** (`agents/synthesis_agent.py`)
Master agent that combines insights from all specialized agents.

**Input Structure - SynthesisInput** (`agents/synthesis_agent.py:26-37`):
```python
@dataclass
class SynthesisInput:
    symbol: str
    sec_analysis: Optional[Dict]
    fundamental_analysis: Optional[Dict]
    technical_analysis: Optional[Dict]
    sentiment_analysis: Optional[Dict]
    peer_comparison: Optional[Dict]
    market_context: Optional[Dict]
    context: Optional[Dict]
    timestamp: datetime
```

**Weight Distribution** (`agents/synthesis_agent.py:18-23`):
```python
SEC = 0.30           # 30% - Regulatory & filing analysis
FUNDAMENTAL = 0.35  # 35% - Financial health
TECHNICAL = 0.20    # 20% - Price action
SENTIMENT = 0.15    # 15% - Market sentiment
```

**Decision Thresholds** (`agents/synthesis_agent.py:66-72`):
```
Strong Buy:  score >= 80
Buy:         score >= 65
Hold:        score >= 50
Sell:        score >= 35
Strong Sell: score < 35
```

#### **SECAnalysisAgent** (`agents/sec_agent.py`)
Extracts and analyzes SEC filing data.

**Key Methods:**
- Fetches company facts from SEC EDGAR
- Extracts 10-K/10-Q submission data
- Analyzes business segments
- Evaluates risk factors
- Identifies regulatory issues

#### **ETFMarketContextAgent** (`agents/etf_market_context_agent.py`)
Provides market context using ETF and macro data.

**Key Methods:**
- Fetches market-wide indicators
- Analyzes sector rotation
- Evaluates macro conditions
- Calculates relative strength vs indices

### 1.3 Agent Manager & Orchestrator

#### **AgentManager** (`agents/manager.py`)
Manages agent lifecycle and task queue.

**Responsibilities:**
- Register/unregister agents
- Queue task management
- Worker task processing
- Agent health monitoring
- Load balancing across agents

**Key Data Structures**:
```python
@dataclass
class ManagedAgent:
    agent: InvestmentAgent
    agent_id: str
    agent_type: str
    status: str  # idle, busy
    current_task: Optional[str]
    completed_tasks: int
    failed_tasks: int
    total_execution_time: float
    last_activity: datetime
```

**Worker Pattern** (`agents/manager.py:235-345`):
- N worker coroutines process task queue
- Find suitable agent for task
- Execute task with error handling
- Update agent stats
- Emit completion/failure events

#### **AgentOrchestrator** (`agents/orchestrator.py`)
High-level coordinator for complex analysis workflows.

**Architecture Diagram** (`agents/orchestrator.py:72-86`):
```
┌─────────────┐   enqueue()    ┌─────────────────────┐   spawn tasks   ┌─────────────┐
│ CLI / API   │ ──────────────▶│ AgentOrchestrator   │ ───────────────▶│ Agent Worker│
└─────┬───────┘                │  • priority queue   │                 │  coroutine  │
      │                        │  • dep graph (DAG)  │◀────────────────└────┬────────┘
      │                        │  • cache & metrics  │    results/emit       │
      │                        └──────────┬──────────┘                      │
      │                                   │                                 │
      │                                   ▼                                 │
      │                          ┌────────────────┐                         │
      └──────────────────────────│  Event Bus     │◀────────────────────────┘
                                 └────────────────┘
```

**Analysis Modes**:
```python
QUICK = "quick"                 # Technical only
STANDARD = "standard"           # Technical + Fundamental
COMPREHENSIVE = "comprehensive" # All agents
CUSTOM = "custom"               # User-defined
```

**Dependency Graph** (`agents/orchestrator.py:154-167`):
```
SEC → Synthesis
Technical → Synthesis
Fundamental → Synthesis
```
Synthesis depends on all other agents and runs last.

**Execution Strategy**:
1. Level 1: Run all independent agents in parallel (SEC, Technical, Fundamental, Market Context)
2. Level 2: Run Synthesis agent with results from Level 1
3. Return aggregated results

**Key Methods**:
- `async analyze(symbol, mode)`: Submit analysis task
- `async analyze_batch(symbols, mode)`: Multiple symbols
- `async analyze_peer_group(target, peers)`: Peer comparison
- `get_status(task_id)`: Check task progress
- `get_results(task_id, wait)`: Retrieve results

---

## 2. Cache System Architecture

The cache system is the **backbone of performance**, enabling rapid analysis by reusing expensive computations.

### 2.1 Cache Type Definitions (`utils/cache/cache_types.py`)

```python
class CacheType(Enum):
    SEC_RESPONSE = "sec_response"              # SEC API responses
    LLM_RESPONSE = "llm_response"              # LLM analysis outputs
    TECHNICAL_DATA = "technical_data"         # Time-series OHLCV data
    SUBMISSION_DATA = "submission_data"       # SEC submissions (RDBMS only)
    COMPANY_FACTS = "company_facts"           # Company financial facts
    QUARTERLY_METRICS = "quarterly_metrics"   # Quarterly summary metrics
    MARKET_CONTEXT = "market_context"         # Market & macro data
```

### 2.2 Cache Manager (`utils/cache/cache_manager.py`)

Central coordinator for multi-layer caching with comprehensive logging and metrics.

**High-Level Flow**:
```
CacheManager
├── register_handler(cache_type, handler)
│   └── Handlers sorted by priority (highest first)
│
├── get(cache_type, key) → data or None
│   └── Try handlers in priority order
│       ├── Handler 1 (priority=10, highest)
│       ├── Handler 2 (priority=5, medium)
│       └── Handler 3 (priority=0, lowest)
│
├── set(cache_type, key, value) → bool
│   └── Write to all applicable handlers
│       ├── Skip lower priority if exists in higher
│       └── File overwrites, RDBMS upserts
│
└── delete_by_symbol(symbol) → Dict[cache_type → count]
    └── Symbol-based cleanup across all types
```

**Key Statistics Tracked**:
```python
_operation_stats = {
    'hits': 0,              # Successful retrievals
    'misses': 0,            # Failed retrievals
    'writes': 0,            # Successful writes
    'errors': 0,            # Operation errors
    'total_time_ms': 0.0,   # Cumulative time
    'avg_time_ms': 0.0,     # Average operation time
    'handler_performance': {}  # Per-handler metrics
}
```

**TTL Management** (`utils/cache/cache_manager.py:811-936`):
```python
Default TTL by cache type:
- SEC_RESPONSE: 6 hours
- COMPANY_FACTS: 90 days
- TECHNICAL_DATA: 7 days
- SUBMISSION_DATA: 90 days
- QUARTERLY_METRICS: 24 hours

LLM_RESPONSE TTL (based on type):
- fundamental: 30 days
- technical: 7 days
- market_context: 10 hours
- synthesis: 7 days
- unknown: 24 hours (default)
```

### 2.3 Cache Handlers - Multi-Layer Strategy

Each handler is responsible for a specific storage medium.

#### **FileCacheStorageHandler** (`utils/cache/file_cache_handler.py`)
Stores JSON/compressed data on disk.

**Priority**: 10 (highest)
**Use Case**: JSON data (LLM responses, company facts, market context)
**Serialization**: `gzip` compressed JSON
**Storage**: `data/{cache_type}/` directories
**Structure**:
```
data/llm_response_cache/
├── AAPL/
│   ├── fundamental_10-K_2024-Q1.json.gz
│   ├── technical_2024-11.json.gz
│   └── synthesis_2024-Q1.json.gz
├── MSFT/
└── ...
```

**Key Methods**:
- `get(key) → Dict`: Decompress and deserialize
- `set(key, value) → bool`: Serialize and compress
- `exists(key) → bool`: Check file existence
- `delete(key) → bool`: Remove file
- `delete_by_symbol(symbol) → int`: Clean all symbol files

#### **ParquetCacheStorageHandler** (`utils/cache/parquet_cache_handler.py`)
Stores tabular time-series data in Parquet format.

**Priority**: 10 (same as File, but for technical data)
**Use Case**: Technical indicators (OHLCV, RSI, MACD, etc.)
**Storage**: `data/technical_data_cache/` with Parquet files
**Structure**:
```
data/technical_data_cache/
├── AAPL_1d.parquet       # Daily candles
├── AAPL_1w.parquet       # Weekly candles
├── AAPL_1mo.parquet      # Monthly candles
└── AAPL_1y.parquet       # Yearly candles
```

**Key Methods**:
- `get(key) → Dict`: Read parquet and convert to dict
- `set(key, value) → bool`: Convert to DataFrame and save parquet
- `upsert(key, value)`: Update existing or create new

#### **RdbmsCacheStorageHandler** (`utils/cache/rdbms_cache_handler.py`)
Stores data in PostgreSQL database for reliability and querying.

**Priority**: 5 (lower than file, higher than audit)
**Use Case**: All cache types (fallback storage)
**Database Schema**:
```sql
CREATE TABLE llm_responses (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    llm_type VARCHAR(100),
    model VARCHAR(100),
    form_type VARCHAR(50),
    period VARCHAR(50),
    response_hash VARCHAR(64),
    response JSONB,
    metadata JSONB,
    cached_at TIMESTAMP,
    ttl_hours INT,
    expires_at TIMESTAMP,
    
    UNIQUE(symbol, llm_type, form_type, period)
);
```

**Key Features**:
- JSONB storage for flexible schema
- Automatic TTL calculation
- Symbol-based cleanup queries
- Upsert operations (update if exists, insert if not)

### 2.4 Cache Promotion Strategy

When data is found in **lower priority storage**, it's automatically **promoted to higher priority** for faster future access.

**Example Flow**:
```
Request for AAPL fundamental analysis
    ↓
Check FileCacheHandler (priority=10)
    ↓ Not found
Check RdbmsCacheHandler (priority=5)
    ↓ Found!
Automatically copy to FileCacheHandler
    ↓
Return data to caller
```

This creates a **hot data migration pattern** where frequently accessed data naturally gravitates to faster storage.

### 2.5 Cache Configuration (`config.py`)

```python
@dataclass
class CacheConfig:
    enabled: bool = True
    storage_type: CacheStorageType = CacheStorageType.DISK
    priority: int = 1
    
    # Disk cache settings
    disk: DiskCacheSettings = {
        'enabled': True,
        'priority': 10,
        'settings': {
            'base_path': 'data/llm_response_cache',
            'compression': 'gzip'
        }
    }
    
    # RDBMS cache settings
    rdbms: RdbmsCacheSettings = {
        'enabled': True,
        'priority': 5,
        'settings': {
            'host': '${DB_HOST:-localhost}',
            'database': 'sec_database'
        }
    }
```

### 2.6 Key Cache Design Patterns

**1. Write-Through Strategy**:
```python
# Agent generates result
result = await agent.process(task)

# Write to ALL handlers
cache.set(cache_type, key, result)
# Both FileCacheStorageHandler AND RdbmsCacheStorageHandler write
```

**2. Read-From-Highest-Priority Strategy**:
```python
# Try File (priority=10) first
result = file_handler.get(key)
if result: return result

# Then try RDBMS (priority=5)
result = rdbms_handler.get(key)
if result:
    # Promote to File for next time
    file_handler.set(key, result)
    return result

# Not found anywhere
return None
```

**3. Symbol-Based Bulk Cleanup**:
```python
# Clear ALL caches for a symbol
delete_by_symbol('AAPL')

# Queries:
# - File: rm -rf data/*/AAPL/
# - RDBMS: DELETE FROM * WHERE symbol='AAPL'
# - Parquet: rm -rf data/technical_data_cache/AAPL*
```

---

## 3. LLM Integration Architecture

### 3.1 Ollama Client (`core/ollama_client.py`)

Low-level async REST API client for Ollama.

**Supported Models** (`core/ollama_client.py:16-45`):
```python
# Premium reasoning (70B+) - BEST FOR ACCURACY
LLAMA3_3_70B = "llama3.3:70b"  # 70.6B, 128K context (LATEST)

# Large reasoning (30-40B)
DEEPSEEK_R1_32B = "deepseek-r1:32b"  # 32.8B, 128K context
QWEN3_30B = "qwen3:30b"  # 30.5B, 262K context (MoE)

# Medium (8-27B)
GEMMA3_27B = "gemma3:27b"
PHI4_REASONING = "phi4-reasoning:plus"  # 14.7B
LLAMA3_1_8B = "llama3.1:8b"  # 8B, 131K context
```

**Key Methods**:
- `async connect()`: Initialize HTTP session
- `async generate(model, prompt, ...)`: Single request
- `async generate_streaming(model, prompt)`: Streaming response
- `async chat(model, messages)`: Multi-turn conversation
- `async list_models()`: Available models
- `async pull_model(name)`: Download model

### 3.2 Resource-Aware Ollama Pool (`core/resource_aware_pool.py`)

Manages multiple Ollama server endpoints with **VRAM-aware load balancing**.

**Architecture Diagram** (`core/resource_aware_pool.py:8-23`):
```
+------------------+      reserve/release      +-----------------------+
|  Agent Request   | ------------------------> | ResourceAwareOllamaPool|
|  (prompt text)   |                           |  (global view)         |
+------------------+ <------------------------ |                       |
        |                 status updates       +----------+------------+
        |                                                   |
        | acquire model                                     | per-server lock
        v                                                   v
+------------------+     poll /api/ps     +----------------------+
| ServerStatus     | <------------------->| Ollama Server (HTTP) |
| (per endpoint)   |                      +----------------------+
+------------------+
```

**Key Data Structures**:

```python
@dataclass
class ServerStatus:
    url: str
    capacity: ServerCapacity  # Hardware specs
    active_requests: int = 0
    running_models: List[RunningModel] = []
    total_vram_used_gb: float = 0.0
    reserved_ram_gb: float = 0.0  # Pessimistic reservation
    
    @property
    def free_ram_gb(self) -> float:
        """Calculate free RAM (actual + reserved)"""
        return max(0.0, 
            self.capacity.usable_ram_gb 
            - self.total_vram_used_gb 
            - self.reserved_ram_gb)
```

**Load Balancing Strategies**:
```python
enum PoolStrategy:
    ROUND_ROBIN = "round_robin"     # Rotate through servers
    RANDOM = "random"               # Random selection
    LEAST_BUSY = "least_busy"       # Fewest active requests
    MOST_CAPACITY = "most_capacity" # Most free VRAM
```

**Key Methods**:
- `async acquire(model, task)`: Reserve VRAM for task
- `async release(allocation_id)`: Release VRAM
- `async get_best_server()`: Find server with most capacity
- `async update_server_status()`: Poll /api/ps for real VRAM

### 3.3 Dynamic LLM Semaphore (`core/llm_semaphore.py`)

Dynamically adjusts concurrency based on actual VRAM constraints.

**Key Concept**: 
The semaphore doesn't use a fixed concurrency limit. Instead, it **tracks actual VRAM usage** and only permits new tasks when there's enough GPU memory available.

**VRAM Tracking** (`core/llm_semaphore.py:134-141`):
```python
self.active_tasks: Dict[str, Dict] = {}  # Currently running tasks
self.used_vram_gb = 0  # Current VRAM usage
self.queue: List[Dict] = []  # Waiting tasks
self.loaded_models: set[str] = set()  # Models in VRAM
self.active_tasks_per_model: Dict[str, int] = {}  # Tasks per model
```

**Task VRAM Calculation** (`core/llm_semaphore.py:227-281`):
```python
async def _calculate_task_vram(
    model: str,
    task_type: str,
    is_cached: bool = False,
    prompt_tokens: Optional[int] = None,
    response_tokens: Optional[int] = None,
    context_tokens: Optional[int] = None
) -> float:
    """
    Calculate VRAM for task:
    - First task: Full model (weights + KV cache)
    - Concurrent tasks: Only KV cache (weights already loaded)
    - Cached tasks: 60% less resources
    """
```

**Acquire Flow** (`core/llm_semaphore.py:287-392`):
```python
async def acquire(
    model: str,
    task_type: str = "summary",
    is_cached: bool = False,
    task_id: str = None
) -> str:
    required_vram = self._calculate_task_vram(model, task_type, is_cached)
    
    if self._can_accommodate_task(required_vram):
        # Can run immediately
        self.active_tasks[allocation_id] = task_info
        self.used_vram_gb += required_vram
        self._stats['total_requests'] += 1
        return allocation_id
    else:
        # Queue and wait
        self.queue.append(task_info)
        self._stats['queue_waits'] += 1
        
        # Wait for resources
        while True:
            await asyncio.sleep(0.1)
            if self._can_accommodate_task(required_vram):
                # Try again
                ...
```

**Statistics** (`core/llm_semaphore.py:433-453`):
```python
def get_stats(self) -> dict:
    return {
        'total_requests': self._stats['total_requests'],
        'current_concurrent': len(self.active_tasks),
        'peak_concurrent': self._stats['concurrent_peak'],
        'queue_size': len(self.queue),
        'queue_waits': self._stats['queue_waits'],
        'cache_hits': self._stats['cache_hits'],
        'cache_hit_rate': ...,
        'vram_used': self.used_vram_gb,
        'vram_available': self.available_vram_gb,
        'vram_utilization': ...,
        'vram_peak': self._stats['vram_peak'],
        'avg_vram_per_task': ...,
        'active_tasks': [task_ids]
    }
```

### 3.4 LLM Patterns (`patterns/llm/`)

Design pattern implementations for clean LLM integration.

#### **LLM Facade** (`patterns/llm/llm_facade.py`)

Provides high-level simplified interface using multiple design patterns.

**Methods by Pattern Type**:

**Template Method Pattern** (High-level analysis):
```python
def analyze_fundamental(symbol, quarterly_data, filing_data):
    """Fundamental analysis using template method"""
    return self.analysis_template.analyze(
        symbol, data, LLMTaskType.FUNDAMENTAL_ANALYSIS
    )

def analyze_technical(symbol, price_data, indicators):
    """Technical analysis using template method"""
    return self.analysis_template.analyze(
        symbol, data, LLMTaskType.TECHNICAL_ANALYSIS
    )

def synthesize_analysis(symbol, fundamental, technical):
    """Synthesize fundamental and technical"""
    return self.analysis_template.analyze(
        symbol, data, LLMTaskType.SYNTHESIS
    )
```

**Strategy Pattern** (Direct LLM operations):
```python
def generate_response(task_type, data):
    """Generate using strategy pattern"""
    request = self.strategy.prepare_request(task_type, data)
    response = self.processor.process_request(request)
    return self.strategy.process_response(response, task_type)
```

**Legacy Compatibility**:
```python
def query_ollama(model, prompt, system_prompt=None, **kwargs):
    """Backward compatible direct Ollama query"""
    
def generate(model, prompt, system_prompt=None, **kwargs):
    """Backward compatible - returns just response string"""
```

#### **LLM Processor** (`patterns/llm/llm_processors.py`)

Chain of Responsibility pattern for request processing.

**Handler Chain**:
```
LLMCacheHandler (check cache)
    ↓
LLMValidationHandler (validate input)
    ↓
LLMStrategyHandler (apply strategy)
    ↓
LLMExecutionHandler (call Ollama)
    ↓
LLMResponseProcessor (parse/normalize)
    ↓
Return result
```

**Queued Processor**:
- Multi-threaded queue processor
- Handles concurrent requests
- Caches at processor level
- Observer pattern for notifications

---

## 4. Data Flow Architecture

### 4.1 End-to-End Analysis Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ User initiates analysis                                           │
│ python cli_orchestrator.py analyze AAPL -m standard              │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. AgentOrchestrator.analyze(symbol, mode)                       │
│    - Create OrchestrationTask with task_id                       │
│    - Determine agents to run based on mode (STANDARD)            │
│    - Queue task with priority                                    │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. Worker picks up task, resolves dependencies                   │
│    - Level 1: SEC, Technical, Fundamental, Market Context        │
│    - Level 2: Synthesis (depends on Level 1)                     │
└────────────────────┬─────────────────────────────────────────────┘
                     │
        ┌────────────┴──────────────┐
        │ Run in parallel           │
        ▼                           ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│ 3a. TechnicalAnalysis   │  │ 3b. FundamentalAnalysis │
│                         │  │                         │
│ ├─ Fetch OHLCV data     │  │ ├─ Fetch SEC facts      │
│ ├─ Compute indicators   │  │ ├─ Calculate ratios     │
│ ├─ Query LLM            │  │ ├─ Extract Q data       │
│ └─ Cache result         │  │ └─ Generate analysis    │
└────────┬────────────────┘  └────────┬────────────────┘
         │                             │
         └────────────┬────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. Synthesis agent combines all inputs                           │
│    - Receives: technical_result, fundamental_result, ...         │
│    - Applies weights: SEC(30%), Fund(35%), Tech(20%), Sent(15%)  │
│    - Generates investment thesis                                 │
│    - Produces overall score (0-100)                              │
│    - Makes decision: STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL        │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. Cache final results                                           │
│    - CacheManager.set(LLM_RESPONSE, key, results)                │
│    - Write to FileCacheHandler AND RdbmsCacheHandler             │
│    - Set TTL based on analysis type                              │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ 6. Return results to user                                        │
│    {                                                              │
│      'symbol': 'AAPL',                                            │
│      'overall_score': 75,                                        │
│      'recommendation': 'BUY',                                    │
│      'investment_thesis': '...',                                │
│      'technical': {...},                                         │
│      'fundamental': {...},                                       │
│      'risks': [...],                                             │
│      'timestamp': '2024-11-02T...'                               │
│    }                                                              │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Cache Lookup Flow

```
Agent task created
        │
        ▼
agent.run(task)
        │
        ▼
Check cache? (has cache_manager)
        │
        ▼
cache_manager.get(cache_type, key)
        │
        ├─────────────────────────────────────┐
        │                                     │
        ▼ Check File first (priority=10)      │
   FileCacheHandler.get()                     │
        │                                     │
        ├─ HIT: Decompress & return          │ MISS
        │                                     │
        └─────────────────────────┬───────────┘
                                  │
                                  ▼
                    RdbmsCacheHandler.get()
                                  │
                                  ├─ HIT: Promote to File
                                  │        (for next time)
                                  │        Return
                                  │
                                  └─ MISS: Process task
                                           (run LLM)
                                           cache result
                                           return
```

### 4.3 SEC Data Extraction Flow

```
symbol (e.g., AAPL)
        │
        ▼
TickerCIKMapper.get_cik(symbol)
        │ → CIK (e.g., 320193)
        │
        ▼
SEC API: /cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K
        │
        ▼ Returns list of filings
        │
        ├─ 10-K (annual)
        ├─ 10-Q (quarterly)
        └─ ...
        │
        ▼
For each filing:
        │
        ├─ Fetch /Archives/.../0000320193-24-047000-index.json
        │  (submission index)
        │
        └─ For each document in submission:
             │
             ├─ Get filing text
             ├─ Parse XBRL/HTML
             ├─ Extract key facts
             └─ Cache in SEC_RESPONSE cache
```

---

## 5. Key Design Patterns

### 5.1 Agent Pattern

**Abstract Base Class with Template Methods**
```python
class InvestmentAgent(ABC):
    async def run(self, task):
        """Template method for task execution"""
        if not await self.pre_process(task):
            return failure_result
        
        result = await self.execute_with_retry(task)
        
        return await self.post_process(task, result)
    
    @abstractmethod
    async def process(self, task):
        """Concrete agents implement this"""
```

### 5.2 Multi-Layer Cache Pattern

**Priority-Based Handler Selection**
```python
handlers = [
    FileCacheHandler(priority=10),      # Fastest
    RdbmsCacheHandler(priority=5),      # Medium
]

for handler in sorted(handlers, key=priority, reverse=True):
    result = handler.get(key)
    if result is not None:
        return result

return None
```

### 5.3 VRAM-Aware Semaphore Pattern

**Dynamic Resource Allocation**
```python
class DynamicLLMSemaphore:
    async def acquire(model, task_type, is_cached):
        required_vram = self._calculate_task_vram(...)
        
        if self.available_vram >= required_vram:
            self.used_vram += required_vram
            return allocation_id
        else:
            # Queue and wait
            while self.available_vram < required_vram:
                await asyncio.sleep(0.1)
            # Allocate when ready
```

### 5.4 Facade Pattern (LLM Operations)

**Simple Interface, Complex Implementation**
```python
class LLMFacade:
    def analyze_fundamental(symbol, data):
        """High-level, simplified interface"""
        return self.analysis_template.analyze(
            symbol, data, LLMTaskType.FUNDAMENTAL_ANALYSIS
        )
    
    # Behind the scenes:
    # - Strategy pattern for request preparation
    # - Chain of Responsibility for handler chain
    # - Observer pattern for notifications
```

### 5.5 Strategy Pattern (LLM Execution)

**Different Strategies for Different Scenarios**
```python
class ComprehensiveLLMStrategy(ILLMStrategy):
    def prepare_request(task_type, data):
        """Prepare detailed, comprehensive request"""
        
class QuickLLMStrategy(ILLMStrategy):
    def prepare_request(task_type, data):
        """Prepare minimal, fast request"""
```

### 5.6 Observer Pattern (Event Bus)

**Decoupled Communication Between Agents**
```python
event_bus.subscribe('analysis_completed', on_analysis_complete)
event_bus.subscribe('agent_failed', on_agent_failure)

# Later:
await event_bus.emit('analysis_completed', {
    'task_id': task_id,
    'symbol': symbol,
    'score': 85
})
```

### 5.7 Dependency Graph Pattern (Orchestrator)

**Topological Sorting for Task Dependencies**
```
build_dependency_graph()
    ├─ SEC → Synthesis
    ├─ Technical → Synthesis
    ├─ Fundamental → Synthesis
    └─ Market Context → Synthesis

get_execution_order(agents):
    Level 0: [SEC, Technical, Fundamental, Market Context] (parallel)
    Level 1: [Synthesis] (depends on Level 0)
```

---

## 6. System Integration Points

### 6.1 Configuration System

**Config Flow**:
```
config.json
    ↓
config.py (Config dataclass)
    ├─ OllamaConfig (models, servers, VRAM specs)
    ├─ SECConfig (API keys, rate limits)
    ├─ CacheConfig (storage backends, TTL)
    ├─ DatabaseConfig (PostgreSQL connection)
    └─ AnalysisConfig (weights, thresholds)
    ↓
get_config() → singleton Config instance
    ↓
Agents, LLM components, Cache system
```

### 6.2 Database Integration

**PostgreSQL Connections**:
```
CacheManager → RdbmsCacheHandler
              ├─ llm_responses table
              ├─ technical_data table
              └─ submission_data table

FundamentalAnalysisAgent → SEC company facts DAO
                        ├─ company_facts table
                        └─ quarterly_metrics table

Utils → ticker_cik_mapper
      └─ ticker_to_cik mapping
```

### 6.3 Event Bus Integration

**Inter-Agent Events**:
```
Agent A completes task
    ↓
emit('analysis_completed', {...})
    ↓
Event Bus queues event
    ↓
Orchestrator listens
    ↓
Trigger dependent tasks
```

### 6.4 Monitoring & Metrics

**Metrics Collection**:
```
Agent execution
    ├─ Task success/failure
    ├─ Processing time
    └─ Cache hit rate
    ↓
MetricsCollector
    ├─ Record to logs
    ├─ Send to monitoring system
    └─ Update dashboards

LLM Semaphore
    ├─ VRAM usage
    ├─ Queue length
    └─ Concurrency level
```

---

## 7. Error Handling & Resilience

### 7.1 Task Retry Logic

```python
async def execute_with_retry(task):
    while task.retry_count <= task.max_retries:
        try:
            result = await execute_with_timeout(task)
            if result.is_successful():
                return result
        except Exception as e:
            task.retry_count += 1
            wait_time = 2 ** task.retry_count  # Exponential backoff
            await asyncio.sleep(wait_time)
    
    return failure_result
```

### 7.2 Cache Fallback

```python
# If cache fails to read, proceed with fresh analysis
try:
    result = cache_manager.get(cache_type, key)
except Exception as e:
    logger.warning(f"Cache read failed: {e}")
    result = None  # Proceed with fresh analysis
```

### 7.3 Graceful Degradation

- Missing VRAM data → use estimates
- Failed SEC lookup → use cached data if available
- Missing technical data → skip pattern analysis
- Synthesis with partial inputs → use available data

---

## 8. Performance Characteristics

### 8.1 Cache Hit Scenarios

**Typical Cache Hit Rates**:
- Fresh symbol first run: 0% (all fresh)
- Subsequent same day: 80-90% (most cached)
- Next day: 10-20% (only long-TTL data like facts)
- After 1 week: Technical data refreshes
- After 30 days: LLM responses refresh

### 8.2 Concurrency Limits

**VRAM-Based (not fixed count)**:
- 48GB GPU with 32B model (32GB) → ~1 concurrent task
- 48GB GPU with 8B model (8GB) → ~4-5 concurrent tasks (with caching)
- Multiple servers → scales linearly

### 8.3 Processing Times

**Typical Durations** (first run):
- Technical analysis: 5-15 seconds
- Fundamental analysis: 10-30 seconds
- Synthesis: 3-10 seconds
- Total analysis: 30-60 seconds

**Cached Run**: <2 seconds (cache retrieval only)

---

## 9. Key Files & Their Roles

| File | Role | Key Responsibility |
|------|------|-------------------|
| `agents/base.py` | Foundation | Agent lifecycle, task execution, caching |
| `agents/orchestrator.py` | Coordinator | Multi-agent workflow, dependency resolution |
| `agents/manager.py` | Lifecycle | Agent registration, task queue, workers |
| `agents/fundamental_agent.py` | Specialist | SEC data extraction, valuation analysis |
| `agents/technical_agent.py` | Specialist | Price analysis, pattern recognition |
| `agents/synthesis_agent.py` | Master | Integration, scoring, recommendations |
| `utils/cache/cache_manager.py` | Core | Cache coordination, TTL, stats |
| `utils/cache/*_handler.py` | Implementation | File/Parquet/RDBMS storage |
| `core/ollama_client.py` | LLM Interface | REST API to Ollama |
| `core/resource_aware_pool.py` | Resource Mgmt | Multi-server load balancing |
| `core/llm_semaphore.py` | Resource Control | VRAM-aware concurrency |
| `patterns/llm/llm_facade.py` | Abstraction | Simplified LLM interface |
| `patterns/llm/llm_processors.py` | Processing | Chain of responsibility handlers |
| `config.py` | Configuration | Single source of truth for settings |
| `utils/event_bus.py` | Communication | Async event system |

---

## Summary

InvestiGator is a **sophisticated, multi-agent investment analysis system** built on:

1. **Agent Architecture**: Specialized agents with common base class for fundamental, technical, SEC, and synthesis analysis
2. **Multi-Layer Caching**: File/Parquet + RDBMS with priority-based retrieval and automatic promotion
3. **Dynamic Resource Management**: VRAM-aware semaphores preventing GPU overload
4. **Design Patterns**: Facade, Strategy, Observer, Dependency Graph, Chain of Responsibility
5. **Event-Driven Communication**: Async event bus for decoupled agent coordination
6. **Comprehensive Error Handling**: Retries, timeouts, cache fallbacks, graceful degradation

The result is a **fast, reliable, and scalable** system that can analyze stocks comprehensively while respecting GPU constraints and maximizing performance through intelligent caching.
