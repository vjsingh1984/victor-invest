# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InvestiGator is an AI-powered investment analysis platform that combines SEC financial data, technical indicators, and multi-agent AI synthesis to provide stock evaluations. It uses Clean Architecture with domain-driven design, built on Victor AI Framework for workflow orchestration.

## Common Commands

```bash
# Install dependencies
pip install -e .                    # Runtime only
pip install -e ".[dev,viz,jupyter]" # Full dev environment
make dev-install                     # Alternative via Makefile

# Run analysis (CLI)
investigator analyze single AAPL --mode quick          # Technical only (~5s)
investigator analyze single AAPL --mode standard       # Tech + Fundamental (~30s)
investigator analyze single AAPL --mode comprehensive  # Full analysis (~60s)
investigator analyze batch AAPL MSFT GOOGL --parallel 4

# Victor-powered CLI (recommended)
victor-invest analyze AAPL --mode comprehensive

# Testing
pytest tests/ -v                    # All tests
pytest tests/ -v -m unit            # Unit tests only
pytest tests/ -v -m "not slow"      # Skip slow tests
pytest tests/ -v -m "not llm"       # Skip tests requiring Ollama
pytest tests/unit/domain/test_agents.py -v  # Single file

# Code quality (via Makefile)
make format                         # Black + isort
make lint                           # Flake8
make type-check                     # mypy
make pre-commit                     # All checks
make ci                             # CI pipeline (format-check + lint + type-check + test-cov)

# Cache management
investigator cache sizes
investigator cache clear --symbol AAPL
investigator cache inspect --symbol AAPL --verbose
```

## Architecture

```
src/investigator/                    # Legacy analysis engine (Clean Architecture)
├── domain/           # Core business logic (no external deps)
│   ├── agents/       # SECAgent, TechnicalAgent, FundamentalAgent, SynthesisAgent
│   ├── models/       # Domain objects (analysis, recommendation)
│   ├── services/     # Valuation (DCF, P/E, GGM), RL policy, data sources
│   └── value_objects/
├── application/      # Use case orchestration
│   ├── orchestrator.py    # Main workflow coordinator
│   └── synthesizer.py     # Multi-agent synthesis (~280KB)
├── infrastructure/   # External integrations
│   ├── cache/        # Multi-layer: File/Parquet + RDBMS
│   ├── database/     # SQLAlchemy models, PostgreSQL/SQLite
│   ├── llm/          # Ollama client, multi-server pool
│   ├── sec/          # SEC EDGAR API integration
│   └── external/     # yfinance, FRED, alternative sources
└── interfaces/cli/   # CLI commands

victor_invest/                       # Victor AI Framework integration
├── workflows/        # YAML-defined analysis workflows
│   ├── quick.yaml              # Technical analysis only (~5s)
│   ├── standard.yaml           # Technical + Fundamental (~30s)
│   ├── comprehensive.yaml      # Full institutional-grade (~60s)
│   ├── peer_comparison.yaml    # Relative valuation analysis
│   └── rl_backtest.yaml        # Historical RL training data
├── handlers.py               # Compute node handlers (shared across workflows)
├── escape_hatches.py         # Conditions/transforms for YAML workflows
├── tools/                    # Victor tool implementations
│   ├── sec_filing.py         # SEC data fetching
│   ├── market_data.py        # Price/market data
│   ├── technical_indicators.py # RSI, MACD, etc.
│   ├── valuation.py          # DCF, P/E, GGM models
│   └── ...
├── agents/specs/             # Agent specifications
├── vertical/                 # Investment vertical definition
└── cli.py                    # Victor-powered CLI
```

**Architecture Patterns:**

1. **Dual Execution Paths:**
   - **Legacy Path**: `investigator` CLI → AgentOrchestrator → multi-agent synthesis
   - **Victor Path**: `victor-invest` CLI → YAML workflows → handlers/tools

2. **Victor Workflow Architecture (Recommended):**
   - YAML-first workflow definitions in `victor_invest/workflows/*.yaml`
   - Compute handlers in `victor_invest/handlers.py` (registered once, used everywhere)
   - Escape hatches in `victor_invest/escape_hatches.py` for complex conditions/transforms
   - Tools in `victor_invest/tools/` for data operations

3. **Context-Stuffing Pattern:**
   - Phase 1-2: Direct tool/handler calls (deterministic, no LLM)
   - Phase 3: Single LLM inference with all collected data (context stuffing)

**Data Flow (Victor Path):**
1. CLI → InvestmentWorkflowProvider.run_agentic_workflow()
2. YAML workflow execution → parallel compute nodes
3. Handlers call tools directly (no orchestrator overhead)
4. Optional LLM synthesis node at end (context-stuffing)
5. Output: WorkflowResult with analysis data

## Key Patterns

**Data Access (Legacy):** Use `DataSourceFacade` for unified data access:
```python
from investigator.domain.services.data_sources.facade import get_data_source_facade
facade = get_data_source_facade()
data = await facade.get_analysis_data("AAPL")
```

**Victor Tool Usage (Recommended):** Direct tool invocation in handlers:
```python
from victor_invest.tools.sec_filing import SECFilingTool

sec_tool = SECFilingTool()
result = await sec_tool.execute({}, symbol="AAPL", action="get_company_facts")
```

**Cache Key Generation:** Must include `fiscal_period` to ensure proper cache hits:
```python
cache_key = SHA256({symbol, analysis_type, context_keys, fiscal_period})
```

**Valuation Models:** DCF, P/E, P/S, EV/EBITDA, GGM, plus sector-specific models for banks, insurance, biotech, semiconductors, REITs. Weights determined by RL policy engine (configurable in config.yaml).

## Configuration

- `config.yaml` - Main configuration (database, LLM, analysis settings, ~2300 lines)
- `.env` - Environment variables (DB credentials, API keys)
- Environment variables override config.yaml

Key env vars: `STOCK_DB_*`, `SEC_DB_*`, `OLLAMA_HOST_*`, `DATABASE_URL`

## Test Markers

- `@pytest.mark.unit` - Fast unit tests (no external dependencies)
- `@pytest.mark.integration` - Requires DB/API
- `@pytest.mark.slow` - Long-running tests (may take several minutes)
- `@pytest.mark.llm` - Requires Ollama
- `@pytest.mark.db` - Requires database
- `@pytest.mark.cache` - Cache-related tests
- `@pytest.mark.performance` - Performance/benchmark tests
- `@pytest.mark.comprehensive` - High coverage tests

## Victor Framework Integration

The project uses Victor's BaseYAMLWorkflowProvider pattern for workflow execution. This provides:
- **YAML-defined workflows** with declarative node definitions
- **Shared handlers** for compute operations (registered once, used everywhere)
- **Escape hatches** for complex conditions/transforms in Python

### Workflow Architecture

```
victor_invest/
├── workflows/
│   ├── __init__.py          # InvestmentWorkflowProvider extends BaseYAMLWorkflowProvider
│   ├── quick.yaml            # Technical analysis only (~5s)
│   ├── standard.yaml         # Technical + Fundamental (~30s)
│   ├── comprehensive.yaml    # Full analysis with LLM synthesis (~60s)
│   ├── peer_comparison.yaml  # Relative valuation analysis
│   ├── rl_backtest.yaml      # RL training data generation
│   └── graphs.py             # Python fallback (StateGraph-based, legacy)
├── handlers.py               # Compute node handlers (shared across workflows)
├── escape_hatches.py         # CONDITIONS & TRANSFORMS for complex logic
├── tools/                    # Victor tool implementations
├── agents/specs/             # Agent specifications for LLM nodes
└── vertical/                 # Investment vertical definition
```

### Using InvestmentWorkflowProvider

```python
from victor_invest.workflows import InvestmentWorkflowProvider

provider = InvestmentWorkflowProvider()

# List available workflows
print(provider.get_workflow_names())  # ['quick', 'standard', 'comprehensive', 'peer_comparison', 'rl_backtest']

# Agentic execution (with LLM support)
result = await provider.run_agentic_workflow(
    "comprehensive",
    context={"symbol": "AAPL"},
    provider="ollama",
    model="gpt-oss:20b",
)
if result.success:
    synthesis = result.context.get("synthesis")
    print(f"Recommendation: {synthesis.get('recommendation')}")

# Compute-only execution (no LLM, faster)
result = await provider.run_workflow_with_handlers(
    "standard",
    context={"symbol": "AAPL"},
)
```

### Handlers vs Escape Hatches vs Tools

- **Handlers** (`victor_invest/handlers.py`): Compute node implementations for YAML workflows
  - `FetchSECDataHandler`, `FundamentalAnalysisHandler`, etc.
  - Registered via `register_handlers()` at module load time
  - Referenced in YAML: `handler: fetch_sec_data`
  - Call tools directly (context-stuffing pattern)

- **Escape Hatches** (`victor_invest/escape_hatches.py`): Python functions for conditions/transforms
  - `data_quality_check`, `valuation_confidence_check`, etc.
  - Used in condition nodes: `condition: "data_quality_check"`
  - Returns branch name ("high", "low", etc.)

- **Tools** (`victor_invest/tools/`): Victor tool implementations for data operations
  - `SECFilingTool`, `MarketDataTool`, `TechnicalIndicatorsTool`, `ValuationTool`, etc.
  - Called by handlers, not directly by YAML
  - Implement `execute()` method with proper error handling

### YAML Workflow Node Types

```yaml
nodes:
  - id: fetch_data
    type: compute              # Calls handler
    handler: fetch_sec_data
    constraints:
      llm_allowed: false
      timeout: 60

  - id: check_quality
    type: condition            # Uses escape hatch
    condition: "data_quality_check"
    branches:
      "high": synthesize
      "low": request_review

  - id: parallel_analysis
    type: parallel             # Parallel execution
    parallel_nodes: [fundamental, technical]
    join_strategy: all
```

## Code Style

- **Line length**: 120 (black default)
- **Type hints**: Required on public functions
- **Async-first**: Python 3.11+ with async/await
- **Formatting**: Black + isort (enforced via Makefile)
- **Linting**: Flake8 for code style
- **Type checking**: mypy for type safety (configured in pyproject.toml)
- **Python versions**: 3.11, 3.12, 3.13 supported

## Development Workflow

When adding new features to Victor workflows:

1. **Add tools** in `victor_invest/tools/` (if new data operations needed)
2. **Add handlers** in `victor_invest/handlers.py` (compute node logic)
3. **Register handlers** via `register_handlers()` call
4. **Define workflow** in `victor_invest/workflows/*.yaml` (YAML-first)
5. **Add escape hatches** in `victor_invest/escape_hatches.py` (only if needed for complex logic)
6. **Write tests** in `tests/victor_invest/` with appropriate markers

**Important**: Always use the context-stuffing pattern:
- Phase 1-2: Direct tool calls (deterministic, no LLM)
- Phase 3: Single LLM call with all context (if needed)

## Special Architectural Features

**Multi-Layer Cache System:**
- File cache (priority 10) - Fastest, JSON-based
- Parquet cache for technical data - Columnar storage
- RDBMS cache (priority 5) - PostgreSQL/SQLite persistent storage
- Automatic promotion between layers based on access patterns

**VRAM-Aware Resource Management:**
- Dynamic LLM semaphore based on actual GPU memory availability
- Multi-server Ollama pool with load balancing
- Model weight calculation for optimal concurrent task scheduling

**Reinforcement Learning:**
- Adaptive model weighting based on historical prediction accuracy
- Contextual bandit policy for model selection
- A/B testing capabilities for model comparison
