# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InvestiGator is an AI-powered investment analysis platform that combines SEC financial data, technical indicators, and multi-agent AI synthesis to provide stock evaluations. It uses Clean Architecture with domain-driven design.

## Common Commands

```bash
# Install dependencies
pip install -e .                    # Runtime only
pip install -e ".[dev,viz,jupyter]" # Full dev environment

# Run analysis
investigator analyze single AAPL --mode standard
investigator analyze batch AAPL MSFT GOOGL --parallel 4

# Testing
pytest tests/ -v                    # All tests
pytest tests/ -v -m unit            # Unit tests only
pytest tests/ -v -m "not slow"      # Skip slow tests
pytest tests/unit/domain/test_agents.py -v  # Single file

# Code quality
make format                         # Black + isort
make lint                           # Flake8
make type-check                     # mypy
make pre-commit                     # All checks

# Cache management
investigator cache sizes
investigator cache clear --symbol AAPL
```

## Architecture

```
src/investigator/
├── domain/           # Core business logic (no external deps)
│   ├── agents/       # SECAgent, TechnicalAgent, FundamentalAgent, SynthesisAgent
│   ├── models/       # Domain objects (analysis, recommendation)
│   ├── services/     # Valuation (DCF, P/E, GGM), RL policy, data sources
│   └── value_objects/
├── application/      # Use case orchestration
│   ├── orchestrator.py    # Main workflow coordinator
│   └── synthesizer.py     # Multi-agent synthesis (~280KB)
├── infrastructure/   # External integrations
│   ├── cache/        # Multi-layer: Memory → File → DB → External
│   ├── database/     # SQLAlchemy models, PostgreSQL/SQLite
│   ├── llm/          # Ollama client, multi-server pool
│   ├── sec/          # SEC EDGAR API integration
│   └── external/     # yfinance, FRED, alternative sources
└── interfaces/cli/   # CLI commands
```

**Data Flow:**
1. CLI → AgentOrchestrator.run_analysis(symbol, mode)
2. Parallel data fetch: SEC filings, market data, technical indicators
3. Parallel LLM analysis per agent
4. Synthesis: weighted multi-model valuation with RL-based model weighting
5. Output: JSON analysis + DB persistence

## Key Patterns

**Data Access:** Always use `DataSourceFacade` for unified data access:
```python
from investigator.domain.services.data_sources.facade import get_data_source_facade
facade = get_data_source_facade()
data = await facade.get_analysis_data("AAPL")
```

**Cache Key Generation:** Must include `fiscal_period` to ensure proper cache hits:
```python
cache_key = SHA256({symbol, analysis_type, context_keys, fiscal_period})
```

**Valuation Models:** DCF, P/E, P/S, EV/EBITDA, GGM, plus sector-specific models for banks, insurance, biotech. Weights determined by RL policy engine.

## Configuration

- `config.yaml` - Main configuration (database, LLM, analysis settings)
- `.env` - Environment variables (DB credentials, API keys)
- Environment variables override config.yaml

Key env vars: `STOCK_DB_*`, `SEC_DB_*`, `OLLAMA_HOST_*`

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Requires DB/API
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.llm` - Requires Ollama
- `@pytest.mark.db` - Requires database

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
│   ├── standard.yaml         # Standard analysis workflow (no LLM)
│   ├── comprehensive.yaml    # Full analysis with LLM synthesis
│   ├── rl_backtest.yaml      # RL training data generation
│   └── graphs.py             # Python fallback (StateGraph-based)
├── handlers.py               # Compute node handlers (shared across CLI, batch, RL)
└── escape_hatches.py         # CONDITIONS & TRANSFORMS for complex logic
```

### Using InvestmentWorkflowProvider

```python
from victor_invest.workflows import InvestmentWorkflowProvider

provider = InvestmentWorkflowProvider()

# List available workflows
print(provider.get_workflow_names())  # ['standard', 'comprehensive', 'rl_backtest', ...]

# Streaming execution (CLI)
async for chunk in provider.astream("comprehensive", orchestrator, {"symbol": "AAPL"}):
    print(f"[{chunk.progress:.0f}%] {chunk.node_name}")

# Non-streaming execution (batch)
from victor.workflows.executor import WorkflowExecutor, WorkflowContext
workflow = provider.get_workflow("standard")
executor = WorkflowExecutor(orchestrator=None)  # No LLM for pure compute
result = await executor.execute(workflow, WorkflowContext({"symbol": "AAPL"}))
```

### Handlers vs Escape Hatches

- **Handlers** (`handlers.py`): Compute node implementations that execute actual operations
  - `fetch_sec_data`, `run_fundamental_analysis`, `run_synthesis`, etc.
  - Registered via `register_handlers()` at module load time
  - Referenced in YAML: `handler: fetch_sec_data`

- **Escape Hatches** (`escape_hatches.py`): Python functions for conditions/transforms
  - `data_quality_check`, `valuation_confidence_check`, etc.
  - Used in condition nodes: `condition: "data_quality_check"`
  - Returns branch name ("high", "low", etc.)

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

- Line length: 120 (black default)
- Type hints required on public functions
- Async-first design (Python 3.11+)
