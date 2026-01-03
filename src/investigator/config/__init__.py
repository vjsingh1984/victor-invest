"""
Configuration Layer

Application configuration with environment variable support.
"""

from investigator.config.config import DatabaseConfig, ModelSpec, OllamaConfig, SECConfig, get_config
from investigator.config.settings import (
    AppSettings,
    CacheControlSettings,
    DatabaseSettings,
    InvestiGatorConfig,
    MonitoringSettings,
    OllamaSettings,
    SECSettings,
    get_settings,
    settings,
)

# Backward compatibility alias
CacheSettings = CacheControlSettings

__all__ = [
    # Legacy config (dataclasses)
    "DatabaseConfig",
    "OllamaConfig",
    "SECConfig",
    "ModelSpec",
    "get_config",
    # New Pydantic settings
    "AppSettings",
    "DatabaseSettings",
    "OllamaSettings",
    "SECSettings",
    "CacheSettings",
    "MonitoringSettings",
    "get_settings",
    "settings",
]
