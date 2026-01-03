"""
Unit tests for configuration settings.

Updated for Phase 4: Nested InvestiGatorConfig structure.
"""

import pytest

from investigator.config import settings


class TestSettings:
    """Test Pydantic settings configuration."""

    def test_settings_initialization(self):
        """Test settings can be loaded."""
        assert settings is not None

    def test_settings_has_required_fields(self):
        """Test settings has required nested configuration sections."""
        required_sections = ["application", "database", "sec", "ollama", "cache_control"]

        for section in required_sections:
            assert hasattr(settings, section), f"Settings missing section: {section}"

    def test_application_name_and_version(self):
        """Test application settings have name and version."""
        assert hasattr(settings.application, "name")
        assert hasattr(settings.application, "version")
        assert settings.application.name == "InvestiGator"

    def test_database_connection_params(self):
        """Test database settings have required connection parameters."""
        assert hasattr(settings.database, "host")
        assert hasattr(settings.database, "port")
        assert hasattr(settings.database, "database")
        assert 1 <= settings.database.port <= 65535

    def test_cache_control_settings(self):
        """Test cache control settings exist."""
        assert hasattr(settings, "cache_control")
        assert hasattr(settings.cache_control, "disk_structure")
        assert hasattr(settings.cache_control.disk_structure, "base_paths")

    def test_debug_is_boolean(self):
        """Test debug flag is boolean in application settings."""
        assert hasattr(settings.application, "debug")
        assert isinstance(settings.application.debug, bool)


class TestSettingsValidation:
    """Test settings validation logic."""

    def test_settings_immutable(self):
        """Test settings are immutable after initialization."""
        # Pydantic settings should prevent modification
        # This would test frozen=True behavior
        pass

    def test_environment_variable_override(self):
        """Test environment variables can override defaults."""
        # Would test .env file loading
        # Would test os.environ override
        pass
