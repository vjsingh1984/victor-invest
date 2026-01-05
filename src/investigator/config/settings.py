"""
Pydantic settings models for InvestiGator configuration.

This module provides type-safe configuration with validation using Pydantic.
Configuration is loaded from config.yaml with environment variable substitution.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# =============================================================================
# Application Settings
# =============================================================================


class ApplicationSettings(BaseSettings):
    """Application metadata and environment configuration."""

    model_config = SettingsConfigDict(env_prefix="APP_")

    name: str = Field(default="InvestiGator")
    version: str = Field(default="0.1.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is development or production."""
        if v not in ["development", "production"]:
            raise ValueError("environment must be 'development' or 'production'")
        return v


# =============================================================================
# Database Settings
# =============================================================================


class DatabaseSettings(BaseSettings):
    """Database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="sec_database")
    username: str = Field(default="investigator")
    password: str = Field(default="")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)

    @property
    def url(self) -> str:
        """Generate database URL from settings."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 0 < v <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return v

    @field_validator("pool_size", "max_overflow")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate value is positive."""
        if v <= 0:
            raise ValueError("must be positive")
        return v


# =============================================================================
# SEC Settings
# =============================================================================


class SECSettings(BaseSettings):
    """SEC EDGAR API configuration."""

    model_config = SettingsConfigDict(env_prefix="SEC_")

    user_agent: str = Field(default="InvestiGator/1.0 (research@example.com)")
    base_url: str = Field(default="https://data.sec.gov/")
    rate_limit: int = Field(default=10)
    cache_dir: str = Field(default="data/sec_cache")
    ticker_cache_file: str = Field(default="data/ticker_cik_map.txt")
    max_retries: int = Field(default=3)
    timeout: int = Field(default=30)
    max_periods_to_analyze: int = Field(default=8)
    require_submissions: bool = Field(default=False)
    include_amended_filings: bool = Field(default=True)

    @field_validator("rate_limit")
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        """Validate rate limit is reasonable."""
        if not 1 <= v <= 100:
            raise ValueError("rate_limit must be between 1 and 100 requests/second")
        return v


# =============================================================================
# Ollama Settings
# =============================================================================


class OllamaServerConfig(BaseSettings):
    """Configuration for a single Ollama server."""

    model_config = SettingsConfigDict(extra="allow")

    url: str
    total_ram_gb: int = Field(default=64)
    usable_ram_gb: int = Field(default=48)
    metal: bool = Field(default=True)
    max_concurrent: int = Field(default=3)
    priority: int = Field(default=0)


class ModelSpecConfig(BaseSettings):
    """Specifications for an LLM model."""

    model_config = SettingsConfigDict(extra="allow")

    context_window: int
    parameters: str
    memory_gb: float
    reasoning_score: float
    thinking_capability: bool
    default_num_predict: int
    max_num_predict: int
    architecture: str
    weights_vram_gb: float
    kv_cache_mb_per_1k_tokens: int
    kv_cache_overhead_pct: float


class OllamaSettings(BaseSettings):
    """Ollama LLM configuration."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="allow")

    base_url: str = Field(default="http://localhost:11434")
    keep_alive: int = Field(default=-1)
    timeout: int = Field(default=1800)
    max_retries: int = Field(default=3)
    min_context_size: int = Field(default=4096)
    num_llm_threads: int = Field(default=1)
    pool_strategy: str = Field(default="prefer_remote")

    servers: List[Dict[str, Any]] = Field(default_factory=list)
    models: Dict[str, str] = Field(default_factory=dict)
    model_specs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    num_predict: Dict[str, int] = Field(default_factory=dict)
    use_toon_format: bool = Field(default=False)
    toon_agents: Dict[str, bool] = Field(default_factory=dict)

    @field_validator("pool_strategy")
    @classmethod
    def validate_pool_strategy(cls, v: str) -> str:
        """Validate pool strategy is valid."""
        valid_strategies = ["most_capacity", "prefer_remote", "round_robin"]
        if v not in valid_strategies:
            raise ValueError(f"pool_strategy must be one of {valid_strategies}")
        return v


# =============================================================================
# Cache Control Settings
# =============================================================================


class DiskCompressionConfig(BaseSettings):
    """Disk cache compression configuration."""

    model_config = SettingsConfigDict(extra="allow")

    enabled: bool = Field(default=True)
    algorithm: str = Field(default="gzip")
    level: int = Field(default=9)
    apply_to_all: bool = Field(default=True)
    file_extensions: List[str] = Field(default_factory=list)


class DiskStructureConfig(BaseSettings):
    """Disk cache structure configuration."""

    model_config = SettingsConfigDict(extra="allow")

    use_symbol_directories: bool = Field(default=True)
    compression: DiskCompressionConfig = Field(default_factory=DiskCompressionConfig)
    base_paths: Dict[str, str] = Field(default_factory=dict)
    directory_structure: Dict[str, str] = Field(default_factory=dict)


class CacheControlSettings(BaseSettings):
    """Cache control configuration."""

    model_config = SettingsConfigDict(extra="allow")

    read_from_cache: bool = Field(default=True)
    write_to_cache: bool = Field(default=True)
    force_refresh: bool = Field(default=False)
    force_refresh_symbols: Optional[List[str]] = Field(default=None)

    storage: List[str] = Field(default_factory=list)
    types: List[str] = Field(default_factory=list)
    disk_structure: DiskStructureConfig = Field(default_factory=DiskStructureConfig)


# =============================================================================
# Analysis Settings
# =============================================================================


class AnalysisSettings(BaseSettings):
    """Analysis configuration."""

    model_config = SettingsConfigDict(extra="allow")

    fundamental_weight: float = Field(default=0.6)
    technical_weight: float = Field(default=0.4)
    min_score_for_buy: float = Field(default=7.0)
    max_score_for_sell: float = Field(default=4.0)
    lookback_days: int = Field(default=365)
    min_volume: int = Field(default=1000000)
    max_prompt_tokens: int = Field(default=32768)

    @field_validator("fundamental_weight", "technical_weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Validate weight is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("weight must be between 0 and 1")
        return v


# =============================================================================
# Orchestrator Settings
# =============================================================================


class OrchestratorSettings(BaseSettings):
    """Orchestrator configuration."""

    model_config = SettingsConfigDict(extra="allow")

    max_concurrent_analyses: int = Field(default=5)
    max_concurrent_agents: int = Field(default=10)
    task_dependency_max_retries: int = Field(default=100)
    task_dependency_backoff_seconds: float = Field(default=0.5)

    @field_validator("max_concurrent_analyses", "max_concurrent_agents")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate value is positive."""
        if v <= 0:
            raise ValueError("must be positive")
        return v


# =============================================================================
# Monitoring Settings
# =============================================================================


class MonitoringSettings(BaseSettings):
    """Monitoring configuration."""

    model_config = SettingsConfigDict(extra="allow")

    enabled: bool = Field(default=True)
    export_interval: int = Field(default=60)
    metrics_port: int = Field(default=9090)

    @field_validator("metrics_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 0 < v <= 65535:
            raise ValueError("metrics_port must be between 1 and 65535")
        return v


# =============================================================================
# Email Settings
# =============================================================================


class EmailSettings(BaseSettings):
    """Email notification configuration."""

    model_config = SettingsConfigDict(extra="allow")

    enabled: bool = Field(default=False)
    smtp_server: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    use_tls: bool = Field(default=True)
    username: str = Field(default="")
    password: str = Field(default="")
    from_address: str = Field(default="investigator@example.com")
    recipients: List[str] = Field(default_factory=list)

    @field_validator("smtp_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 0 < v <= 65535:
            raise ValueError("smtp_port must be between 1 and 65535")
        return v


# =============================================================================
# Tracking Settings
# =============================================================================


class TrackingSettings(BaseSettings):
    """Symbol tracking configuration."""

    model_config = SettingsConfigDict(extra="allow")

    symbols: List[str] = Field(default_factory=list)


# =============================================================================
# Vector Database Settings
# =============================================================================


class VectorDBSettings(BaseSettings):
    """Vector database configuration."""

    model_config = SettingsConfigDict(extra="allow")

    enabled: bool = Field(default=False)
    db_path: str = Field(default="data/vector_db")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")


# =============================================================================
# Valuation Settings
# =============================================================================


class ValuationSettings(BaseSettings):
    """Valuation configuration."""

    model_config = SettingsConfigDict(extra="allow")

    sector_multiples_path: str = Field(default="config/sector_multiples.json")
    sector_multiples_freshness_days: int = Field(default=7)
    sector_multiples_delta_threshold: float = Field(default=0.15)
    liquidity_floor_usd: int = Field(default=5000000)
    ggm_payout_threshold_pct: float = Field(default=40.0)

    fading_dcf_thresholds: Dict[str, float] = Field(default_factory=dict)
    sector_normalization: Dict[str, List[str]] = Field(default_factory=dict)
    tier_thresholds: Dict[str, Any] = Field(default_factory=dict)
    tier_base_weights: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    industry_specific_weights: Dict[str, Any] = Field(default_factory=dict)
    model_applicability: Dict[str, Any] = Field(default_factory=dict)
    data_quality_thresholds: Dict[str, Any] = Field(default_factory=dict)
    outlier_detection: Dict[str, Any] = Field(default_factory=dict)
    model_fallback: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# DCF Valuation Settings
# =============================================================================


class DCFValuationSettings(BaseSettings):
    """DCF valuation configuration."""

    model_config = SettingsConfigDict(extra="allow")

    sector_based_parameters: Dict[str, Any] = Field(default_factory=dict)
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    wacc_parameters: Dict[str, Any] = Field(default_factory=dict)
    fcf_growth_parameters: Dict[str, Any] = Field(default_factory=dict)
    fcf_growth_caps_by_sector: Dict[str, Any] = Field(default_factory=dict)
    rule_of_40: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Main Configuration
# =============================================================================


class InvestiGatorConfig(BaseSettings):
    """
    Master configuration - single source of truth.

    This configuration is loaded from config.yaml with environment variable
    substitution and validated using Pydantic.

    Example:
        >>> config = InvestiGatorConfig.from_yaml("config.yaml")
        >>> print(config.database.url)
        postgresql://investigator:***@${DB_HOST:-localhost}:5432/sec_database
    """

    model_config = SettingsConfigDict(extra="allow")

    application: ApplicationSettings = Field(default_factory=ApplicationSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    sec: SECSettings = Field(default_factory=SECSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    cache_control: CacheControlSettings = Field(default_factory=CacheControlSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    valuation: ValuationSettings = Field(default_factory=ValuationSettings)
    dcf_valuation: DCFValuationSettings = Field(default_factory=DCFValuationSettings)

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
            ValueError: If required environment variable is missing

        Example:
            >>> config = InvestiGatorConfig.from_yaml("config.yaml")
            >>> print(config.database.url)
            postgresql://investigator:***@${DB_HOST:-localhost}:5432/sec_database
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load YAML
        with open(config_path, "r") as f:
            yaml_content = f.read()

        # Substitute environment variables
        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default_value}
        def env_var_replacer(match):
            var_spec = match.group(1)
            if ":-" in var_spec:
                var_name, default = var_spec.split(":-", 1)
                return os.getenv(var_name, default)
            else:
                value = os.getenv(var_spec)
                if value is None:
                    raise ValueError(f"Environment variable {var_spec} not set and no default provided")
                return value

        yaml_content = re.sub(r"\$\{([^}]+)\}", env_var_replacer, yaml_content)

        # Parse YAML
        config_dict = yaml.safe_load(yaml_content)

        # Validate and create instance
        return cls(**config_dict)


# =============================================================================
# Backward Compatibility
# =============================================================================

# Alias for backward compatibility
AppSettings = InvestiGatorConfig


# Backward compatibility function
def get_settings() -> InvestiGatorConfig:
    """
    Get application settings singleton (backward compatibility).

    Returns:
        InvestiGatorConfig: Application settings

    Example:
        >>> settings = get_settings()
        >>> print(settings.database.url)
    """
    return InvestiGatorConfig.from_yaml("config.yaml")


# Singleton instance (backward compatibility)
try:
    settings = get_settings()
except (FileNotFoundError, ValueError) as e:
    # Fallback to defaults if config.yaml doesn't exist yet or has issues
    # This allows the module to load even if config isn't set up yet
    settings = InvestiGatorConfig()


__all__ = [
    "InvestiGatorConfig",
    "AppSettings",
    "ApplicationSettings",
    "DatabaseSettings",
    "SECSettings",
    "OllamaSettings",
    "CacheControlSettings",
    "AnalysisSettings",
    "OrchestratorSettings",
    "MonitoringSettings",
    "EmailSettings",
    "TrackingSettings",
    "VectorDBSettings",
    "ValuationSettings",
    "DCFValuationSettings",
    "get_settings",
    "settings",
]
