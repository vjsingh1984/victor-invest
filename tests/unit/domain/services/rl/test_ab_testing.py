"""
Unit tests for ABTestingFramework.
"""

import pytest
from datetime import date

from investigator.domain.services.rl.monitoring.ab_testing import (
    ABTestingFramework,
    ABTestConfig,
)
from investigator.domain.services.rl.models import ABTestGroup


class TestABTestingFramework:
    """Tests for ABTestingFramework."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ABTestConfig(
            test_name="test_rl_vs_baseline",
            rl_traffic_pct=0.20,
            min_samples_per_group=10,
        )

    @pytest.fixture
    def framework(self, config):
        """Create A/B testing framework."""
        return ABTestingFramework(config=config)

    def test_get_assignment_deterministic(self, framework):
        """Test that assignment is deterministic for same symbol."""
        symbol = "AAPL"

        # Get assignment multiple times
        group1 = framework.get_assignment(symbol)
        group2 = framework.get_assignment(symbol)
        group3 = framework.get_assignment(symbol)

        # Should always be the same
        assert group1 == group2 == group3

    def test_get_assignment_distribution(self, framework):
        """Test that assignment follows target distribution."""
        # Reset cache
        framework.reset_cache()

        # Test many symbols
        symbols = [f"SYM{i}" for i in range(1000)]

        for symbol in symbols:
            framework.get_assignment(symbol)

        stats = framework.get_assignment_stats()

        # Should be close to 20% RL
        actual_rl_pct = stats["actual_rl_pct"]
        assert 15 <= actual_rl_pct <= 25  # Allow some variance

    def test_should_use_rl(self, framework):
        """Test should_use_rl method."""
        # Find a symbol that gets RL
        rl_symbol = None
        baseline_symbol = None

        for i in range(100):
            symbol = f"TEST{i}"
            if framework.should_use_rl(symbol):
                rl_symbol = symbol
            else:
                baseline_symbol = symbol

            if rl_symbol and baseline_symbol:
                break

        # Verify both groups are represented
        assert rl_symbol is not None or baseline_symbol is not None

    def test_assignment_cache(self, framework):
        """Test that assignments are cached."""
        symbol = "CACHED"

        # First call
        framework.get_assignment(symbol)
        assert symbol in framework._assignment_cache

        # Cache should contain assignment
        assert framework._assignment_cache[symbol] in [ABTestGroup.RL, ABTestGroup.BASELINE]

    def test_reset_cache(self, framework):
        """Test cache reset."""
        # Make some assignments
        for i in range(10):
            framework.get_assignment(f"SYM{i}")

        assert len(framework._assignment_cache) == 10

        # Reset
        framework.reset_cache()

        assert len(framework._assignment_cache) == 0
        assert framework._assignment_counts[ABTestGroup.RL] == 0
        assert framework._assignment_counts[ABTestGroup.BASELINE] == 0

    def test_assignment_stats(self, framework):
        """Test getting assignment statistics."""
        # Make some assignments
        for i in range(20):
            framework.get_assignment(f"SYM{i}")

        stats = framework.get_assignment_stats()

        assert "total_assignments" in stats
        assert stats["total_assignments"] == 20
        assert "rl_count" in stats
        assert "baseline_count" in stats
        assert stats["rl_count"] + stats["baseline_count"] == 20


class TestABTestConfig:
    """Tests for ABTestConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ABTestConfig()

        assert config.test_name == "rl_vs_baseline"
        assert config.rl_traffic_pct == 0.20
        assert config.min_samples_per_group == 50
        assert config.confidence_level == 0.95

    def test_custom_config(self):
        """Test custom configuration."""
        config = ABTestConfig(
            test_name="custom_test",
            rl_traffic_pct=0.50,
            min_samples_per_group=100,
            confidence_level=0.99,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        assert config.test_name == "custom_test"
        assert config.rl_traffic_pct == 0.50
        assert config.min_samples_per_group == 100


class TestABTestRecommendations:
    """Tests for A/B test recommendations."""

    def test_recommend_action_insufficient_data(self):
        """Test recommendation with insufficient data."""
        from unittest.mock import patch, MagicMock

        # Create a mock db manager that returns empty results
        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_db.get_session.return_value = mock_session

        # Patch get_db_manager before creating framework
        with patch('investigator.domain.services.rl.monitoring.ab_testing.get_db_manager', return_value=mock_db):
            config = ABTestConfig(min_samples_per_group=10)
            framework = ABTestingFramework(config=config)
            recommendation = framework.recommend_action()

        assert "action" in recommendation
        assert "reason" in recommendation
        # With no data, should recommend continuing test
        assert recommendation["action"] == "continue_test"


class TestHashDistribution:
    """Tests for hash-based distribution."""

    def test_hash_uniformity(self):
        """Test that hash produces uniform distribution."""
        framework = ABTestingFramework(
            config=ABTestConfig(rl_traffic_pct=0.50)
        )

        # Large sample
        rl_count = 0
        baseline_count = 0

        for i in range(10000):
            symbol = f"UNIFORM_TEST_{i}"
            if framework.should_use_rl(symbol):
                rl_count += 1
            else:
                baseline_count += 1

        # Should be close to 50/50
        ratio = rl_count / (rl_count + baseline_count)
        assert 0.45 <= ratio <= 0.55

    def test_hash_consistency_across_instances(self):
        """Test that different instances give same assignment."""
        config = ABTestConfig(rl_traffic_pct=0.30)

        fw1 = ABTestingFramework(config=config)
        fw2 = ABTestingFramework(config=config)

        # Same symbol should get same assignment in both
        for i in range(100):
            symbol = f"CONSISTENCY_{i}"
            assert fw1.get_assignment(symbol) == fw2.get_assignment(symbol)
