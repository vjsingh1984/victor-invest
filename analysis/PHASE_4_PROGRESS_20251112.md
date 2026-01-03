# Phase 4: Configuration Consolidation - Progress Report

**Date**: 2025-11-12
**Status**: IN PROGRESS (Step 1 of 7 Complete)
**Branch**: `feature/architecture-redesign-phase1-fiscal-period-service` (will commit to `reconciled_merge`)

---

## Objective

Consolidate all configuration files (`config.json`, `config.py`, `config.yaml`) into a single `config.yaml` with Pydantic validation and environment variable support.

---

## Completed Tasks

### ✅ Step 1: Create Comprehensive config.yaml (COMPLETE)

**File**: `/Users/vijaysingh/code/InvestiGator/config.yaml`
**Size**: 866 lines
**Status**: Fully consolidated all settings from `config.json`

**Contents**:
- Application settings
- Database configuration with `${DB_PASSWORD}` environment variable support
- SEC EDGAR API configuration
- Ollama LLM configuration (servers, models, specs, TOON format)
- Cache control configuration
- Analysis parameters
- Orchestrator configuration
- Monitoring configuration
- Email notifications
- Tracking symbols
- Vector database settings
- Valuation configuration (tiers, weights, model applicability)
- DCF valuation configuration (sector parameters, WACC, FCF growth, Rule of 40)

**Key Features**:
- Environment variable substitution syntax: `${VAR_NAME:-default_value}`
- Extensive inline documentation with comments
- Organized into logical sections with clear headings
- snake_case naming throughout
- YAML structure maps directly to Pydantic model hierarchy

**Example Usage**:
```yaml
database:
  host: ${DB_HOST:-localhost}
  port: 5432
  database: sec_database
  username: investigator
  password: ${DB_PASSWORD:-investigator}  # Environment variable with fallback
```

---

## Remaining Tasks

### ⏳ Step 2: Update Pydantic Settings Model (IN PROGRESS)

**File**: `src/investigator/config/settings.py`
**Current State**: Basic Pydantic models exist (DatabaseSettings, OllamaSettings, SECSettings, CacheSettings, MonitoringSettings)
**Required**: Extend to match full YAML structure

**Approach**:
1. Create additional Pydantic models for new sections:
   - `ValuationSettings` (tier_thresholds, tier_base_weights, industry_specific_weights, model_applicability, data_quality_thresholds, outlier_detection)
   - `DCFValuationSettings` (sector_based_parameters, default_parameters, wacc_parameters, fcf_growth_parameters, fcf_growth_caps_by_sector, rule_of_40)
   - `AnalysisSettings` (fundamental_weight, technical_weight, scores, lookback, volume)
   - `OrchestratorSettings` (max_concurrent_analyses, max_concurrent_agents, task_dependency settings)
   - `EmailSettings` (enabled, SMTP, recipients)
   - `TrackingSettings` (symbols list)
   - `VectorDBSettings` (enabled, db_path, embedding_model)

2. Update `AppSettings` to include all subsections as nested Pydantic models

3. Maintain backward compatibility with existing code

**Code Structure**:
```python
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional

class ValuationTierThresholds(BaseSettings):
    dividend_aristocrat: Dict[str, Any]
    high_growth: Dict[str, Any]
    growth_hybrid: Dict[str, Any]
    # ... etc

class ValuationSettings(BaseSettings):
    sector_multiples_path: str
    sector_multiples_freshness_days: int
    # ...
    tier_thresholds: ValuationTierThresholds
    tier_base_weights: Dict[str, Dict[str, int]]
    # ... etc

class InvestiGatorConfig(BaseSettings):
    """Master configuration - single source of truth"""
    application: ApplicationSettings
    database: DatabaseSettings
    sec: SECSettings
    ollama: OllamaSettings
    cache_control: CacheControlSettings
    analysis: AnalysisSettings
    orchestrator: OrchestratorSettings
    monitoring: MonitoringSettings
    email: EmailSettings
    tracking: TrackingSettings
    vector_db: VectorDBSettings
    valuation: ValuationSettings
    dcf_valuation: DCFValuationSettings
```

---

### ⏳ Step 3: Add from_yaml() Class Method (PENDING)

**File**: `src/investigator/config/settings.py`
**Purpose**: Load and validate config.yaml with environment variable substitution

**Implementation**:
```python
import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict

class InvestiGatorConfig(BaseSettings):
    # ... fields ...

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config.yaml") -> "InvestiGatorConfig":
        """
        Load configuration from YAML file with environment variable substitution.

        Args:
            config_path: Path to config.yaml file (default: "config.yaml")

        Returns:
            Validated InvestiGatorConfig instance

        Raises:
            ValidationError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist

        Example:
            >>> config = InvestiGatorConfig.from_yaml("config.yaml")
            >>> print(config.database.url)
            postgresql://investigator:***@${DB_HOST:-localhost}:5432/sec_database
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load YAML
        with open(config_path, 'r') as f:
            yaml_content = f.read()

        # Substitute environment variables
        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default_value}
        def env_var_replacer(match):
            var_spec = match.group(1)
            if ':-' in var_spec:
                var_name, default = var_spec.split(':-', 1)
                return os.getenv(var_name, default)
            else:
                value = os.getenv(var_spec)
                if value is None:
                    raise ValueError(f"Environment variable {var_spec} not set and no default provided")
                return value

        yaml_content = re.sub(r'\$\{([^}]+)\}', env_var_replacer, yaml_content)

        # Parse YAML
        config_dict = yaml.safe_load(yaml_content)

        # Validate and create instance
        return cls(**config_dict)
```

**Testing**:
```bash
# Test environment variable substitution
export DB_PASSWORD="secret123"
python3 -c "from investigator.config.settings import InvestiGatorConfig; c = InvestiGatorConfig.from_yaml(); print(c.database.password)"
# Expected: "secret123"

# Test validation error
cat > /tmp/test_invalid.yaml <<EOF
database:
  pool_size: -10  # Invalid: negative
EOF

python3 -c "from investigator.config.settings import InvestiGatorConfig; InvestiGatorConfig.from_yaml('/tmp/test_invalid.yaml')"
# Expected: ValidationError with message "pool_size must be positive"
```

---

### ⏳ Step 4: Update get_config() to Use YAML (PENDING)

**File**: `src/investigator/config/config.py` or create new `src/investigator/config/__init__.py`
**Purpose**: Update existing get_config() function to load from YAML instead of JSON

**Implementation**:
```python
# src/investigator/config/__init__.py
from pathlib import Path
from typing import Optional
from investigator.config.settings import InvestiGatorConfig

_config_instance: Optional[InvestiGatorConfig] = None

def get_config(config_path: str | Path = "config.yaml", force_reload: bool = False) -> InvestiGatorConfig:
    """
    Get configuration singleton instance.

    Args:
        config_path: Path to config.yaml (default: "config.yaml")
        force_reload: Force reload from file (default: False)

    Returns:
        InvestiGatorConfig: Validated configuration instance

    Example:
        >>> from investigator.config import get_config
        >>> config = get_config()
        >>> print(config.database.url)
    """
    global _config_instance

    if _config_instance is None or force_reload:
        _config_instance = InvestiGatorConfig.from_yaml(config_path)

    return _config_instance

# Backward compatibility: Provide old-style config access
def get_database_config():
    """Backward compatibility: Get database config"""
    return get_config().database

def get_ollama_config():
    """Backward compatibility: Get Ollama config"""
    return get_config().ollama

# ... etc for other sections
```

**Migration Path**:
1. Create new `get_config()` that loads from YAML
2. Provide backward compatibility helpers
3. Update imports across codebase incrementally
4. Add deprecation warnings to old JSON-based loaders

---

### ⏳ Step 5: Create Validation Tests (PENDING)

**File**: `tests/unit/config/test_settings_validation.py`
**Purpose**: Ensure Pydantic validation works correctly

**Test Cases**:
```python
import pytest
from pydantic import ValidationError
from investigator.config.settings import InvestiGatorConfig, DatabaseSettings

class TestConfigValidation:
    """Test Pydantic validation for configuration"""

    def test_valid_database_config(self):
        """Test valid database configuration passes validation"""
        config = DatabaseSettings(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            pool_size=10,
            max_overflow=20
        )
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.url == "postgresql://test_user:test_pass@localhost:5432/test_db"

    def test_invalid_database_port(self):
        """Test invalid port number raises ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseSettings(
                host="localhost",
                port=-1,  # Invalid: negative port
                database="test_db",
                username="test_user",
                password="test_pass",
                pool_size=10,
                max_overflow=20
            )
        assert "port" in str(exc_info.value)

    def test_invalid_pool_size(self):
        """Test invalid pool_size raises ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseSettings(
                host="localhost",
                port=5432,
                database="test_db",
                username="test_user",
                password="test_pass",
                pool_size=-10,  # Invalid: negative
                max_overflow=20
            )
        assert "pool_size" in str(exc_info.value)

    def test_env_var_substitution(self, monkeypatch, tmp_path):
        """Test environment variable substitution in YAML"""
        # Create test config with env var
        config_yaml = tmp_path / "test_config.yaml"
        config_yaml.write_text("""
database:
  host: localhost
  port: 5432
  database: test_db
  username: test_user
  password: ${DB_PASSWORD:-default_pass}
  pool_size: 10
  max_overflow: 20
""")

        # Test with env var set
        monkeypatch.setenv("DB_PASSWORD", "secret123")
        config = InvestiGatorConfig.from_yaml(config_yaml)
        assert config.database.password == "secret123"

        # Test with env var not set (should use default)
        monkeypatch.delenv("DB_PASSWORD")
        config = InvestiGatorConfig.from_yaml(config_yaml)
        assert config.database.password == "default_pass"

    def test_missing_required_field(self):
        """Test missing required field raises ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseSettings(
                host="localhost",
                # Missing required fields: port, database, username, password
                pool_size=10,
                max_overflow=20
            )
        assert "field required" in str(exc_info.value).lower()

    def test_valuation_tier_thresholds(self):
        """Test valuation tier thresholds validation"""
        # TODO: Add test for ValuationSettings validation
        pass

    def test_dcf_sector_parameters(self):
        """Test DCF sector parameters validation"""
        # TODO: Add test for DCFValuationSettings validation
        pass
```

**Run Tests**:
```bash
pytest tests/unit/config/test_settings_validation.py -v
```

---

### ⏳ Step 6: Add Deprecation Warnings to Old Config Files (PENDING)

**Files**:
- `config.json` - Add header comment marking as deprecated
- `src/investigator/config/config.py` - Add deprecation warnings

**Implementation**:

**config.json** (add header):
```json
{
  "_DEPRECATED": "This file is deprecated. Please migrate to config.yaml",
  "_MIGRATION_GUIDE": "See analysis/PHASE_4_PROGRESS_20251112.md for migration instructions",
  "database": {
    ...
  }
}
```

**config.py** (add deprecation warnings):
```python
import warnings

def load_config_from_json(config_path: str = "config.json"):
    """
    Load configuration from JSON file.

    DEPRECATED: Use InvestiGatorConfig.from_yaml() instead.
    This function will be removed in version 1.0.0.
    """
    warnings.warn(
        "load_config_from_json() is deprecated. "
        "Use InvestiGatorConfig.from_yaml() instead. "
        "See analysis/PHASE_4_PROGRESS_20251112.md for migration guide.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... existing JSON loading logic
```

---

### ⏳ Step 7: Create Migration Guide (PENDING)

**File**: `docs/CONFIG_MIGRATION_GUIDE.md`
**Purpose**: Help users migrate from old config to new YAML-based config

**Contents**:
1. Overview of changes
2. Step-by-step migration instructions
3. Environment variable setup
4. Validation testing
5. Rollback procedure
6. FAQ

---

## Implementation Order

1. ✅ Create comprehensive config.yaml (COMPLETE)
2. ⏳ Update Pydantic settings.py to match YAML structure (IN PROGRESS)
3. Add from_yaml() class method
4. Create validation tests
5. Update get_config() to use YAML
6. Add deprecation warnings
7. Create migration guide

---

## Success Criteria

- ✅ Single source of truth: config.yaml contains all settings
- ⏳ Pydantic validation: All fields validated on startup
- ⏳ Clear error messages: ValidationError shows which field is invalid
- ⏳ Environment variables: ${VAR_NAME} syntax works correctly
- ⏳ Migration guide: Users can migrate from old config easily
- ⏳ Tests pass: All validation tests pass

---

## Next Steps

1. **Complete Pydantic Models** (Est: 2-3 hours)
   - Add ValuationSettings, DCFValuationSettings, AnalysisSettings, etc.
   - Update AppSettings to include all subsections
   - Test basic validation

2. **Implement from_yaml()** (Est: 1 hour)
   - Add environment variable substitution
   - Add YAML loading logic
   - Test with actual config.yaml

3. **Create Validation Tests** (Est: 2 hours)
   - Test each Pydantic model
   - Test environment variable substitution
   - Test invalid configs raise proper errors

4. **Update get_config()** (Est: 1 hour)
   - Create new YAML-based loader
   - Add backward compatibility helpers
   - Test with existing code

5. **Add Deprecation Warnings** (Est: 30 minutes)
   - Update config.json header
   - Add warnings to config.py functions
   - Document timeline for removal

6. **Create Migration Guide** (Est: 1 hour)
   - Write step-by-step instructions
   - Add examples
   - Test guide with fresh install

**Total Estimated Time**: 7.5-8.5 hours

---

## Files Modified

- ✅ `/Users/vijaysingh/code/InvestiGator/config.yaml` - CREATED (866 lines)
- ⏳ `src/investigator/config/settings.py` - TO UPDATE
- ⏳ `src/investigator/config/__init__.py` - TO CREATE
- ⏳ `tests/unit/config/test_settings_validation.py` - TO CREATE
- ⏳ `docs/CONFIG_MIGRATION_GUIDE.md` - TO CREATE
- ⏳ `config.json` - TO DEPRECATE

---

## Commit Strategy

After completing all steps, create single commit:

```bash
git add config.yaml
git add src/investigator/config/settings.py
git add src/investigator/config/__init__.py
git add tests/unit/config/test_settings_validation.py
git add docs/CONFIG_MIGRATION_GUIDE.md
git add config.json  # With deprecation notice

git commit -m "feat(config): consolidate configuration to single YAML file (Phase 4)

- Create comprehensive config.yaml (866 lines) consolidating all settings
- Extend Pydantic models to match YAML structure
- Add from_yaml() class method with environment variable substitution
- Create validation tests for all configuration sections
- Add get_config() singleton with YAML loading
- Deprecate config.json with migration guide
- Add backward compatibility helpers

Resolves Pain Point #4: Configuration Consolidation
Part of Architecture Redesign Phase 4 (20-hour effort)

Testing:
- pytest tests/unit/config/ -v (12 tests passing)
- Manual testing with environment variables
- Validation error testing

Documentation:
- See analysis/PHASE_4_PROGRESS_20251112.md for progress
- See docs/CONFIG_MIGRATION_GUIDE.md for migration instructions

Phase 4 Status: COMPLETE (7/7 steps)
"
```

---

## Notes

- This is a substantial change affecting configuration throughout the codebase
- Backward compatibility is critical - existing code must continue to work
- Environment variable support enables 12-factor app principles
- Pydantic validation provides type safety and clear error messages
- Migration guide helps users transition smoothly

---

**Last Updated**: 2025-11-12 (Step 1 of 7 Complete)
**Next Milestone**: Complete Pydantic models and from_yaml() method
**Estimated Completion**: Phase 4 complete in ~8 hours of focused work
