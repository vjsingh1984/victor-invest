#!/usr/bin/env python3
"""
Validation tests for Pydantic configuration models.

Tests configuration validation, environment variable substitution,
and error handling for invalid configurations.
"""

import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from pydantic import ValidationError

from investigator.config.settings import (
    AnalysisSettings,
    ApplicationSettings,
    CacheControlSettings,
    DatabaseSettings,
    DCFValuationSettings,
    EmailSettings,
    InvestiGatorConfig,
    MonitoringSettings,
    OllamaSettings,
    OrchestratorSettings,
    SECSettings,
    TrackingSettings,
    ValuationSettings,
    VectorDBSettings,
)


class TestDatabaseSettings:
    """Test DatabaseSettings validation."""

    def test_valid_database_config(self):
        """Test valid database configuration passes validation."""
        config = DatabaseSettings(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            pool_size=10,
            max_overflow=20,
        )
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.username == "test_user"
        assert config.password == "test_pass"
        assert config.pool_size == 10
        assert config.max_overflow == 20

    def test_database_url_property(self):
        """Test database URL generation."""
        config = DatabaseSettings(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
        )
        expected = "postgresql://test_user:test_pass@localhost:5432/test_db"
        assert config.url == expected

    def test_invalid_port_negative(self):
        """Test negative port number raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseSettings(
                host="localhost",
                port=-1,
                database="test_db",
                username="test_user",
                password="test_pass",
            )
        assert "port must be between 1 and 65535" in str(exc_info.value)

    def test_invalid_port_too_large(self):
        """Test port number > 65535 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseSettings(
                host="localhost",
                port=70000,
                database="test_db",
                username="test_user",
                password="test_pass",
            )
        assert "port must be between 1 and 65535" in str(exc_info.value)

    def test_invalid_pool_size_negative(self):
        """Test negative pool_size raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseSettings(
                host="localhost",
                port=5432,
                database="test_db",
                username="test_user",
                password="test_pass",
                pool_size=-10,
            )
        assert "must be positive" in str(exc_info.value)

    def test_invalid_max_overflow_zero(self):
        """Test zero max_overflow raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseSettings(
                host="localhost",
                port=5432,
                database="test_db",
                username="test_user",
                password="test_pass",
                max_overflow=0,
            )
        assert "must be positive" in str(exc_info.value)


class TestApplicationSettings:
    """Test ApplicationSettings validation."""

    def test_valid_application_config(self):
        """Test valid application configuration."""
        config = ApplicationSettings(
            name="InvestiGator",
            version="0.1.0",
            environment="development",
            debug=True,
        )
        assert config.name == "InvestiGator"
        assert config.version == "0.1.0"
        assert config.environment == "development"
        assert config.debug is True

    def test_invalid_environment(self):
        """Test invalid environment value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ApplicationSettings(environment="staging")
        assert "environment must be 'development' or 'production'" in str(exc_info.value)

    def test_valid_production_environment(self):
        """Test production environment is valid."""
        config = ApplicationSettings(environment="production")
        assert config.environment == "production"


class TestSECSettings:
    """Test SECSettings validation."""

    def test_valid_sec_config(self):
        """Test valid SEC configuration."""
        config = SECSettings(
            user_agent="InvestiGator/1.0 (research@example.com)",
            base_url="https://data.sec.gov/",
            rate_limit=10,
            max_retries=3,
            timeout=30,
        )
        assert config.rate_limit == 10
        assert config.max_retries == 3
        assert config.timeout == 30

    def test_invalid_rate_limit_too_low(self):
        """Test rate limit < 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SECSettings(rate_limit=0)
        assert "rate_limit must be between 1 and 100" in str(exc_info.value)

    def test_invalid_rate_limit_too_high(self):
        """Test rate limit > 100 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SECSettings(rate_limit=150)
        assert "rate_limit must be between 1 and 100" in str(exc_info.value)


class TestOllamaSettings:
    """Test OllamaSettings validation."""

    def test_valid_ollama_config(self):
        """Test valid Ollama configuration."""
        config = OllamaSettings(
            base_url="http://localhost:11434",
            timeout=1800,
            pool_strategy="prefer_remote",
        )
        assert config.base_url == "http://localhost:11434"
        assert config.timeout == 1800
        assert config.pool_strategy == "prefer_remote"

    def test_invalid_pool_strategy(self):
        """Test invalid pool strategy raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            OllamaSettings(pool_strategy="invalid_strategy")
        assert "pool_strategy must be one of" in str(exc_info.value)

    def test_valid_pool_strategies(self):
        """Test all valid pool strategies."""
        valid_strategies = ["most_capacity", "prefer_remote", "round_robin"]
        for strategy in valid_strategies:
            config = OllamaSettings(pool_strategy=strategy)
            assert config.pool_strategy == strategy


class TestAnalysisSettings:
    """Test AnalysisSettings validation."""

    def test_valid_analysis_config(self):
        """Test valid analysis configuration."""
        config = AnalysisSettings(
            fundamental_weight=0.6,
            technical_weight=0.4,
            min_score_for_buy=7.0,
            max_score_for_sell=4.0,
        )
        assert config.fundamental_weight == 0.6
        assert config.technical_weight == 0.4

    def test_invalid_weight_negative(self):
        """Test negative weight raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisSettings(fundamental_weight=-0.5)
        assert "weight must be between 0 and 1" in str(exc_info.value)

    def test_invalid_weight_too_large(self):
        """Test weight > 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisSettings(technical_weight=1.5)
        assert "weight must be between 0 and 1" in str(exc_info.value)


class TestOrchestratorSettings:
    """Test OrchestratorSettings validation."""

    def test_valid_orchestrator_config(self):
        """Test valid orchestrator configuration."""
        config = OrchestratorSettings(
            max_concurrent_analyses=5,
            max_concurrent_agents=10,
        )
        assert config.max_concurrent_analyses == 5
        assert config.max_concurrent_agents == 10

    def test_invalid_max_concurrent_analyses_negative(self):
        """Test negative max_concurrent_analyses raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            OrchestratorSettings(max_concurrent_analyses=-5)
        assert "must be positive" in str(exc_info.value)

    def test_invalid_max_concurrent_agents_zero(self):
        """Test zero max_concurrent_agents raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            OrchestratorSettings(max_concurrent_agents=0)
        assert "must be positive" in str(exc_info.value)


class TestMonitoringSettings:
    """Test MonitoringSettings validation."""

    def test_valid_monitoring_config(self):
        """Test valid monitoring configuration."""
        config = MonitoringSettings(
            enabled=True,
            export_interval=60,
            metrics_port=9090,
        )
        assert config.enabled is True
        assert config.export_interval == 60
        assert config.metrics_port == 9090

    def test_invalid_metrics_port_negative(self):
        """Test negative metrics port raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MonitoringSettings(metrics_port=-1)
        assert "metrics_port must be between 1 and 65535" in str(exc_info.value)

    def test_invalid_metrics_port_too_large(self):
        """Test metrics port > 65535 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MonitoringSettings(metrics_port=70000)
        assert "metrics_port must be between 1 and 65535" in str(exc_info.value)


class TestEmailSettings:
    """Test EmailSettings validation."""

    def test_valid_email_config(self):
        """Test valid email configuration."""
        config = EmailSettings(
            enabled=True,
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            use_tls=True,
            username="user@example.com",
            password="secret",
        )
        assert config.smtp_port == 587
        assert config.use_tls is True

    def test_invalid_smtp_port_negative(self):
        """Test negative SMTP port raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EmailSettings(smtp_port=-1)
        assert "smtp_port must be between 1 and 65535" in str(exc_info.value)


class TestEnvironmentVariableSubstitution:
    """Test environment variable substitution in from_yaml()."""

    def test_env_var_substitution_with_value(self, tmp_path, monkeypatch):
        """Test environment variable substitution when var is set."""
        config_yaml = tmp_path / "test_config.yaml"
        config_yaml.write_text("""
application:
  name: InvestiGator
  version: 0.1.0
  environment: development
  debug: false

database:
  host: localhost
  port: 5432
  database: test_db
  username: test_user
  password: ${DB_PASSWORD:-default_pass}
  pool_size: 10
  max_overflow: 20

sec:
  user_agent: "InvestiGator/1.0 (research@example.com)"
  base_url: "https://data.sec.gov/"
  rate_limit: 10
  cache_dir: "data/sec_cache"
  ticker_cache_file: "data/ticker_cik_map.txt"
  max_retries: 3
  timeout: 30
  max_periods_to_analyze: 8
  require_submissions: false
  include_amended_filings: true

ollama:
  base_url: "http://localhost:11434"
  keep_alive: -1
  timeout: 1800
  max_retries: 3
  min_context_size: 4096
  num_llm_threads: 1
  pool_strategy: "prefer_remote"

cache_control:
  read_from_cache: true
  write_to_cache: true
  force_refresh: false

analysis:
  fundamental_weight: 0.6
  technical_weight: 0.4
  min_score_for_buy: 7.0
  max_score_for_sell: 4.0
  lookback_days: 365
  min_volume: 1000000
  max_prompt_tokens: 32768

orchestrator:
  max_concurrent_analyses: 5
  max_concurrent_agents: 10
  task_dependency_max_retries: 100
  task_dependency_backoff_seconds: 0.5

monitoring:
  enabled: true
  export_interval: 60
  metrics_port: 9090

email:
  enabled: false
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  use_tls: true
  username: ""
  password: ""

tracking:
  symbols: []

vector_db:
  enabled: false
  db_path: "data/vector_db"
  embedding_model: "all-MiniLM-L6-v2"

valuation:
  sector_multiples_path: "config/sector_multiples.json"
  sector_multiples_freshness_days: 7
  sector_multiples_delta_threshold: 0.15
  liquidity_floor_usd: 5000000
  ggm_payout_threshold_pct: 40.0

dcf_valuation:
  sector_based_parameters: {}
  default_parameters: {}
  wacc_parameters: {}
  fcf_growth_parameters: {}
""")

        # Test with env var set
        monkeypatch.setenv("DB_PASSWORD", "secret123")
        config = InvestiGatorConfig.from_yaml(config_yaml)
        assert config.database.password == "secret123"

    def test_env_var_substitution_with_default(self, tmp_path, monkeypatch):
        """Test environment variable substitution when var not set (use default)."""
        config_yaml = tmp_path / "test_config.yaml"
        config_yaml.write_text("""
application:
  name: InvestiGator
  version: 0.1.0
  environment: development
  debug: false

database:
  host: localhost
  port: 5432
  database: test_db
  username: test_user
  password: ${DB_PASSWORD:-default_pass}
  pool_size: 10
  max_overflow: 20

sec:
  user_agent: "InvestiGator/1.0 (research@example.com)"
  base_url: "https://data.sec.gov/"
  rate_limit: 10
  cache_dir: "data/sec_cache"
  ticker_cache_file: "data/ticker_cik_map.txt"
  max_retries: 3
  timeout: 30
  max_periods_to_analyze: 8
  require_submissions: false
  include_amended_filings: true

ollama:
  base_url: "http://localhost:11434"
  keep_alive: -1
  timeout: 1800
  max_retries: 3
  min_context_size: 4096
  num_llm_threads: 1
  pool_strategy: "prefer_remote"

cache_control:
  read_from_cache: true
  write_to_cache: true
  force_refresh: false

analysis:
  fundamental_weight: 0.6
  technical_weight: 0.4
  min_score_for_buy: 7.0
  max_score_for_sell: 4.0
  lookback_days: 365
  min_volume: 1000000
  max_prompt_tokens: 32768

orchestrator:
  max_concurrent_analyses: 5
  max_concurrent_agents: 10
  task_dependency_max_retries: 100
  task_dependency_backoff_seconds: 0.5

monitoring:
  enabled: true
  export_interval: 60
  metrics_port: 9090

email:
  enabled: false
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  use_tls: true
  username: ""
  password: ""

tracking:
  symbols: []

vector_db:
  enabled: false
  db_path: "data/vector_db"
  embedding_model: "all-MiniLM-L6-v2"

valuation:
  sector_multiples_path: "config/sector_multiples.json"
  sector_multiples_freshness_days: 7
  sector_multiples_delta_threshold: 0.15
  liquidity_floor_usd: 5000000
  ggm_payout_threshold_pct: 40.0

dcf_valuation:
  sector_based_parameters: {}
  default_parameters: {}
  wacc_parameters: {}
  fcf_growth_parameters: {}
""")

        # Test with env var not set (should use default)
        monkeypatch.delenv("DB_PASSWORD", raising=False)
        config = InvestiGatorConfig.from_yaml(config_yaml)
        assert config.database.password == "default_pass"

    def test_env_var_substitution_no_default_raises_error(self, tmp_path, monkeypatch):
        """Test environment variable substitution without default raises ValueError."""
        config_yaml = tmp_path / "test_config.yaml"
        config_yaml.write_text("""
application:
  name: InvestiGator
  version: 0.1.0
  environment: development
  debug: false

database:
  host: localhost
  port: 5432
  database: test_db
  username: test_user
  password: ${REQUIRED_DB_PASSWORD}
  pool_size: 10
  max_overflow: 20

sec:
  user_agent: "InvestiGator/1.0 (research@example.com)"
  base_url: "https://data.sec.gov/"
  rate_limit: 10

ollama:
  base_url: "http://localhost:11434"

cache_control:
  read_from_cache: true

analysis:
  fundamental_weight: 0.6
  technical_weight: 0.4

orchestrator:
  max_concurrent_analyses: 5
  max_concurrent_agents: 10

monitoring:
  enabled: true

email:
  enabled: false

tracking:
  symbols: []

vector_db:
  enabled: false

valuation:
  sector_multiples_path: "config/sector_multiples.json"

dcf_valuation:
  sector_based_parameters: {}
""")

        monkeypatch.delenv("REQUIRED_DB_PASSWORD", raising=False)
        with pytest.raises(ValueError) as exc_info:
            InvestiGatorConfig.from_yaml(config_yaml)
        assert "REQUIRED_DB_PASSWORD" in str(exc_info.value)
        assert "not set and no default provided" in str(exc_info.value)


class TestInvestiGatorConfig:
    """Test InvestiGatorConfig master configuration."""

    def test_valid_full_config(self):
        """Test valid full configuration with all sections."""
        config = InvestiGatorConfig(
            application=ApplicationSettings(name="InvestiGator", version="0.1.0"),
            database=DatabaseSettings(
                host="localhost",
                port=5432,
                database="test_db",
                username="test_user",
                password="test_pass",
            ),
            sec=SECSettings(),
            ollama=OllamaSettings(),
            cache_control=CacheControlSettings(),
            analysis=AnalysisSettings(),
            orchestrator=OrchestratorSettings(),
            monitoring=MonitoringSettings(),
            email=EmailSettings(),
            tracking=TrackingSettings(),
            vector_db=VectorDBSettings(),
            valuation=ValuationSettings(),
            dcf_valuation=DCFValuationSettings(),
        )

        assert config.application.name == "InvestiGator"
        assert config.database.host == "localhost"
        assert config.database.port == 5432

    def test_config_defaults(self):
        """Test configuration with default values."""
        config = InvestiGatorConfig()

        # Check defaults
        assert config.application.name == "InvestiGator"
        assert config.database.port == 5432
        assert config.sec.rate_limit == 10
        assert config.ollama.timeout == 1800
        assert config.analysis.fundamental_weight == 0.6
        assert config.orchestrator.max_concurrent_analyses == 5
        assert config.monitoring.metrics_port == 9090

    def test_from_yaml_file_not_found(self, tmp_path):
        """Test from_yaml() raises FileNotFoundError for missing file."""
        nonexistent_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError) as exc_info:
            InvestiGatorConfig.from_yaml(nonexistent_file)
        assert "Configuration file not found" in str(exc_info.value)


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_cache_settings_alias(self):
        """Test CacheSettings alias for backward compatibility."""
        from investigator.config import CacheSettings

        # Should be same as CacheControlSettings
        from investigator.config import CacheControlSettings

        assert CacheSettings is CacheControlSettings

    def test_app_settings_alias(self):
        """Test AppSettings alias for backward compatibility."""
        from investigator.config.settings import AppSettings

        # Should be same as InvestiGatorConfig
        assert AppSettings is InvestiGatorConfig
