# Clean Architecture Migration Guide

Guide for migrating code to the new Clean Architecture structure.

## Quick Reference

### Old Import Patterns → New Import Patterns

```python
# OLD (root imports)
from agents.fundamental_agent import FundamentalAnalysisAgent
from utils.data_normalizer import DataNormalizer
from config import get_config

# NEW (Clean Architecture)
from investigator.domain.agents.fundamental import FundamentalAnalysisAgent
from investigator.domain.services.data_normalizer import DataNormalizer
from investigator.config import settings
```

## Layer-by-Layer Migration

### 1. Domain Layer Imports

**Agents**:
```python
# OLD
from agents.fundamental_agent import FundamentalAnalysisAgent
from agents.technical_agent import TechnicalAnalysisAgent

# NEW
from investigator.domain.agents.fundamental import FundamentalAnalysisAgent
from investigator.domain.agents.technical import TechnicalAnalysisAgent
```

**Models**:
```python
# OLD
from agents.base import AgentTask, AgentResult, AnalysisType

# NEW
from investigator.domain.models.analysis import AgentTask, AgentResult, AnalysisType
```

### 2. Infrastructure Layer Imports

**LLM**:
```python
# OLD
from core.resource_aware_pool import ResourceAwareOllamaPool
from utils.ollama_client import OllamaClient

# NEW
from investigator.infrastructure.llm.pool import ResourceAwareOllamaPool
from investigator.infrastructure.llm.ollama import OllamaClient
```

**Cache**:
```python
# OLD
from utils.cache.cache_manager import CacheManager
from utils.cache.cache_types import CacheType

# NEW
from investigator.infrastructure.cache.cache_manager import CacheManager
from investigator.infrastructure.cache.cache_types import CacheType
```

### 3. Application Layer Imports

```python
# OLD
from agents.orchestrator import AgentOrchestrator, AnalysisMode

# NEW
from investigator.application import AgentOrchestrator, AnalysisMode, AnalysisService
```

### 4. Configuration

```python
# OLD (dataclass-based)
from config import get_config
config = get_config()
db_url = config.database.url

# NEW (Pydantic with env vars)
from investigator.config import settings
db_url = settings.database.url
```

## Running with New Architecture

### Set PYTHONPATH

For development without installing:
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
python3.11 script.py
```

### Install in Editable Mode

For proper package imports:
```bash
python3.11 -m pip install -e .
```

### Environment Variables

Create .env file:
```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=sec_database
DB_USERNAME=investigator
DB_PASSWORD=yourpassword

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=qwen2.5:32b-instruct

# SEC
SEC_USER_AGENT=InvestiGator yourname@example.com

# Cache
CACHE_TTL_LLM=21600
CACHE_TTL_TECHNICAL=86400

# App
APP_ENVIRONMENT=development
APP_DEBUG=false
```

## Common Migration Patterns

### Pattern 1: Service Creation

**OLD**:
```python
orchestrator = AgentOrchestrator(cache, metrics)
await orchestrator.start()
task_id = await orchestrator.analyze("AAPL", AnalysisMode.STANDARD)
results = await orchestrator.get_results(task_id, wait=True)
```

**NEW**:
```python
from investigator.application import AnalysisService

service = AnalysisService(cache, metrics)
await service.start()
results = await service.analyze_stock("AAPL", mode="standard")
```

### Pattern 2: Configuration Access

**OLD**:
```python
config = get_config()
if config.ollama.servers:
    for server in config.ollama.servers:
        print(server.url)
```

**NEW**:
```python
from investigator.config import settings

if settings.ollama.servers:
    for server_url in settings.ollama.servers:
        print(server_url)
```

## Backward Compatibility

The old imports still work during transition:
- `config.py` in root → Use for now, migrate to `investigator.config` later
- `agents/` → Old location still works, new code should use `investigator.domain.agents`
- `utils/` → Most utilities still in old location

## Testing

### Running Tests with New Structure

```bash
# Set PYTHONPATH for pytest
PYTHONPATH=src pytest tests/ -v

# Or use pyproject.toml configuration
pytest tests/ -v
```

### Writing New Tests

```python
import pytest
from investigator.domain.agents.fundamental import FundamentalAnalysisAgent
from investigator.infrastructure.cache import CacheManager

@pytest.mark.asyncio
async def test_fundamental_analysis():
    cache = CacheManager()
    agent = FundamentalAnalysisAgent("test", None, None, cache)
    # ... test code
```

## Troubleshooting

### ModuleNotFoundError: No module named 'investigator'

**Solution**: Set PYTHONPATH or install package
```bash
export PYTHONPATH="${PWD}/src"
# OR
pip install -e .
```

### Import errors from old code

**Solution**: Update imports to new paths
```python
# Change this
from agents.fundamental_agent import FundamentalAnalysisAgent

# To this
from investigator.domain.agents.fundamental import FundamentalAnalysisAgent
```

### Configuration not found

**Solution**: Create .env file or set environment variables
```bash
cp .env.example .env
# Edit .env with your settings
```

## Migration Checklist

- [ ] Update imports to use `investigator.*` namespace
- [ ] Replace `get_config()` with `settings` from `investigator.config`
- [ ] Use `AnalysisService` instead of direct `AgentOrchestrator` access
- [ ] Create .env file for environment-specific config
- [ ] Set PYTHONPATH or install package in editable mode
- [ ] Run tests to verify migration
- [ ] Update documentation

## Next Steps

1. Complete Phase 4: Finish application services
2. Complete Phase 5: Migrate all CLI commands
3. Phase 7: Remove old files after full migration
4. Update all scripts to use new imports
5. Deploy with new architecture

## Support

For questions or issues with migration:
- Check REFACTORING_IMPLEMENTATION_TRACKER.md for status
- Review CLEAN_ARCHITECTURE.md for architecture details
- See examples in src/investigator/ for reference implementations
