"""
Unit tests for weight_audit_trail module.

Tests weight evolution tracking and audit logging.
"""

import json
import pytest
from investigator.domain.services.weight_audit_trail import (
    WeightAuditTrail,
    WeightAdjustment,
    AuditStep,
    AuditSummary,
)


class TestWeightAdjustment:
    """Tests for WeightAdjustment dataclass."""

    def test_creation(self):
        """Test creating a weight adjustment."""
        adj = WeightAdjustment(
            model='dcf',
            source='market_context',
            multiplier=0.90,
            reason='Bearish trend detected'
        )
        assert adj.model == 'dcf'
        assert adj.source == 'market_context'
        assert adj.multiplier == 0.90
        assert 'Bearish' in adj.reason

    def test_creation_minimal(self):
        """Test creating adjustment without optional reason."""
        adj = WeightAdjustment(
            model='pe',
            source='data_quality',
            multiplier=0.95
        )
        assert adj.reason is None


class TestAuditStep:
    """Tests for AuditStep dataclass."""

    def test_creation(self):
        """Test creating an audit step."""
        step = AuditStep(
            step_number=1,
            step_name='base_weights',
            timestamp='2024-01-01T12:00:00',
            weights_before={},
            weights_after={'dcf': 30, 'pe': 25, 'ps': 45},
            adjustments=[]
        )
        assert step.step_number == 1
        assert step.step_name == 'base_weights'
        assert step.weights_after['dcf'] == 30

    def test_get_changes_no_change(self):
        """Test get_changes with no significant change."""
        step = AuditStep(
            step_number=1,
            step_name='test',
            timestamp='2024-01-01T12:00:00',
            weights_before={'dcf': 30.0},
            weights_after={'dcf': 30.0},
            adjustments=[]
        )
        changes = step.get_changes()
        assert len(changes) == 0

    def test_get_changes_with_change(self):
        """Test get_changes detects significant changes."""
        step = AuditStep(
            step_number=2,
            step_name='market_context',
            timestamp='2024-01-01T12:00:00',
            weights_before={'dcf': 30.0, 'pe': 25.0},
            weights_after={'dcf': 24.0, 'pe': 28.0},
            adjustments=[]
        )
        changes = step.get_changes()
        assert 'dcf' in changes
        assert changes['dcf']['before'] == 30.0
        assert changes['dcf']['after'] == 24.0
        assert changes['dcf']['delta'] == -6.0

    def test_get_changes_new_model(self):
        """Test get_changes with new model added."""
        step = AuditStep(
            step_number=1,
            step_name='test',
            timestamp='2024-01-01T12:00:00',
            weights_before={'dcf': 30.0},
            weights_after={'dcf': 30.0, 'pe': 25.0},
            adjustments=[]
        )
        changes = step.get_changes()
        assert 'pe' in changes
        assert changes['pe']['before'] == 0
        assert changes['pe']['after'] == 25.0

    def test_get_changes_model_removed(self):
        """Test get_changes with model removed."""
        step = AuditStep(
            step_number=1,
            step_name='test',
            timestamp='2024-01-01T12:00:00',
            weights_before={'dcf': 30.0, 'pe': 25.0},
            weights_after={'dcf': 30.0},
            adjustments=[]
        )
        changes = step.get_changes()
        assert 'pe' in changes
        assert changes['pe']['after'] == 0


class TestAuditSummary:
    """Tests for AuditSummary dataclass."""

    def test_creation(self):
        """Test creating an audit summary."""
        summary = AuditSummary(
            symbol='AAPL',
            total_steps=3,
            initial_weights={'dcf': 30, 'pe': 25, 'ps': 45},
            final_weights={'dcf': 25, 'pe': 30, 'ps': 45},
            total_adjustments=5,
            bounds_applied=True,
            largest_change={'model': 'pe', 'delta': 5.0},
            warnings=['pe: Weight changed by +20%']
        )
        assert summary.symbol == 'AAPL'
        assert summary.total_steps == 3
        assert summary.bounds_applied
        assert len(summary.warnings) == 1


class TestWeightAuditTrail:
    """Tests for WeightAuditTrail class."""

    @pytest.fixture
    def trail(self):
        """Create a fresh audit trail."""
        return WeightAuditTrail(symbol='AAPL')

    def test_initialization(self, trail):
        """Test trail initialization."""
        assert trail.symbol == 'AAPL'
        assert len(trail.steps) == 0
        assert not trail.bounds_applied

    def test_capture_single_step(self, trail):
        """Test capturing a single step."""
        trail.capture(
            step_number=1,
            step_name='base_weights',
            weights_before={},
            weights_after={'dcf': 30, 'pe': 25, 'ps': 45},
            adjustments=[]
        )
        assert len(trail.steps) == 1
        assert trail.steps[0].step_name == 'base_weights'

    def test_capture_multiple_steps(self, trail):
        """Test capturing multiple steps."""
        # Step 1: Base weights
        trail.capture(
            step_number=1,
            step_name='base_weights',
            weights_before={},
            weights_after={'dcf': 30, 'pe': 25, 'ps': 45},
            adjustments=[]
        )

        # Step 2: Market context
        trail.capture(
            step_number=2,
            step_name='market_context',
            weights_before={'dcf': 30, 'pe': 25, 'ps': 45},
            weights_after={'dcf': 24, 'pe': 22.5, 'ps': 49.5},
            adjustments=[
                WeightAdjustment('dcf', 'trend', 0.8, 'Bearish'),
                WeightAdjustment('pe', 'sentiment', 0.9, 'Neutral'),
            ]
        )

        assert len(trail.steps) == 2
        assert trail.steps[1].step_number == 2
        assert len(trail.steps[1].adjustments) == 2

    def test_capture_with_metadata(self, trail):
        """Test capturing step with metadata."""
        trail.capture(
            step_number=1,
            step_name='test',
            weights_before={},
            weights_after={'dcf': 30},
            adjustments=[],
            metadata={'sector': 'Technology', 'tier': 'high_growth'}
        )
        assert trail.steps[0].metadata['sector'] == 'Technology'

    def test_mark_bounds_applied(self, trail):
        """Test marking bounds as applied."""
        assert not trail.bounds_applied
        trail.mark_bounds_applied()
        assert trail.bounds_applied

    def test_get_summary_empty(self, trail):
        """Test getting summary from empty trail."""
        summary = trail.get_summary()
        assert summary.symbol == 'AAPL'
        assert summary.total_steps == 0
        assert summary.total_adjustments == 0

    def test_get_summary_with_steps(self, trail):
        """Test getting summary with steps."""
        trail.capture(
            step_number=1,
            step_name='base_weights',
            weights_before={},
            weights_after={'dcf': 30, 'pe': 70},
            adjustments=[]
        )
        trail.capture(
            step_number=2,
            step_name='market_context',
            weights_before={'dcf': 30, 'pe': 70},
            weights_after={'dcf': 20, 'pe': 80},
            adjustments=[
                WeightAdjustment('dcf', 'trend', 0.67, 'Bearish'),
                WeightAdjustment('pe', 'trend', 1.14, 'Bullish'),
            ]
        )

        summary = trail.get_summary()
        assert summary.total_steps == 2
        assert summary.total_adjustments == 2
        assert summary.initial_weights == {'dcf': 30, 'pe': 70}
        assert summary.final_weights == {'dcf': 20, 'pe': 80}

    def test_get_summary_largest_change(self, trail):
        """Test summary finds largest change."""
        # Start with existing weights (not from zero)
        trail.capture(
            step_number=1,
            step_name='base',
            weights_before={'dcf': 50, 'pe': 50},  # Start from existing
            weights_after={'dcf': 50, 'pe': 50},   # No change in step 1
            adjustments=[]
        )
        trail.capture(
            step_number=2,
            step_name='adjustment',
            weights_before={'dcf': 50, 'pe': 50},
            weights_after={'dcf': 30, 'pe': 70},  # DCF drops by 20
            adjustments=[]
        )

        summary = trail.get_summary()
        assert summary.largest_change is not None
        # Largest change is either dcf (-20) or pe (+20)
        assert abs(summary.largest_change['delta']) == 20

    def test_get_summary_warnings(self, trail):
        """Test summary generates warnings for large changes."""
        trail.capture(
            step_number=1,
            step_name='base',
            weights_before={},
            weights_after={'dcf': 40, 'pe': 60},
            adjustments=[]
        )
        trail.capture(
            step_number=2,
            step_name='final',
            weights_before={'dcf': 40, 'pe': 60},
            weights_after={'dcf': 10, 'pe': 90},  # DCF drops 75%
            adjustments=[]
        )

        summary = trail.get_summary()
        assert len(summary.warnings) > 0
        assert any('dcf' in w.lower() for w in summary.warnings)

    def test_to_dict(self, trail):
        """Test converting trail to dictionary."""
        trail.capture(
            step_number=1,
            step_name='base',
            weights_before={},
            weights_after={'dcf': 30, 'pe': 70},
            adjustments=[WeightAdjustment('dcf', 'tier', 1.0, 'Base')]
        )

        data = trail.to_dict()
        assert data['symbol'] == 'AAPL'
        assert 'summary' in data
        assert 'steps' in data
        assert len(data['steps']) == 1
        assert data['steps'][0]['step_name'] == 'base'

    def test_to_json(self, trail):
        """Test converting trail to JSON string."""
        trail.capture(
            step_number=1,
            step_name='test',
            weights_before={},
            weights_after={'dcf': 50, 'pe': 50},
            adjustments=[]
        )

        json_str = trail.to_json()
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed['symbol'] == 'AAPL'

    def test_to_json_with_indent(self, trail):
        """Test JSON with custom indent."""
        trail.capture(
            step_number=1,
            step_name='test',
            weights_before={},
            weights_after={'dcf': 100},
            adjustments=[]
        )

        json_str = trail.to_json(indent=4)
        # Should have newlines due to indentation
        assert '\n' in json_str

    def test_get_step_found(self, trail):
        """Test getting step by number."""
        trail.capture(
            step_number=1,
            step_name='first',
            weights_before={},
            weights_after={'dcf': 50},
            adjustments=[]
        )
        trail.capture(
            step_number=2,
            step_name='second',
            weights_before={'dcf': 50},
            weights_after={'dcf': 40},
            adjustments=[]
        )

        step = trail.get_step(2)
        assert step is not None
        assert step.step_name == 'second'

    def test_get_step_not_found(self, trail):
        """Test getting non-existent step."""
        trail.capture(
            step_number=1,
            step_name='only',
            weights_before={},
            weights_after={'dcf': 50},
            adjustments=[]
        )

        step = trail.get_step(99)
        assert step is None

    def test_get_steps_for_model(self, trail):
        """Test getting steps that affected a model."""
        trail.capture(
            step_number=1,
            step_name='base',
            weights_before={},
            weights_after={'dcf': 50, 'pe': 50},
            adjustments=[]
        )
        trail.capture(
            step_number=2,
            step_name='adjust_dcf',
            weights_before={'dcf': 50, 'pe': 50},
            weights_after={'dcf': 40, 'pe': 50},  # Only DCF changes
            adjustments=[]
        )
        trail.capture(
            step_number=3,
            step_name='adjust_pe',
            weights_before={'dcf': 40, 'pe': 50},
            weights_after={'dcf': 40, 'pe': 60},  # Only PE changes
            adjustments=[]
        )

        dcf_steps = trail.get_steps_for_model('dcf')
        assert len(dcf_steps) == 2  # Step 1 (created) and Step 2 (changed)

        pe_steps = trail.get_steps_for_model('pe')
        assert len(pe_steps) == 2  # Step 1 (created) and Step 3 (changed)

    def test_get_steps_for_model_none(self, trail):
        """Test getting steps for model with no changes."""
        trail.capture(
            step_number=1,
            step_name='base',
            weights_before={},
            weights_after={'dcf': 50},
            adjustments=[]
        )

        steps = trail.get_steps_for_model('pe')  # PE not in trail
        assert len(steps) == 0

    def test_log_summary(self, trail, caplog):
        """Test logging summary output."""
        import logging
        caplog.set_level(logging.INFO)

        trail.capture(
            step_number=1,
            step_name='base',
            weights_before={},
            weights_after={'dcf': 50, 'pe': 50},
            adjustments=[]
        )
        trail.mark_bounds_applied()

        trail.log_summary()

        # Check that key elements were logged
        log_text = caplog.text
        assert 'AAPL' in log_text or trail.symbol in str(caplog.records)


class TestWeightAuditTrailEdgeCases:
    """Edge case tests for WeightAuditTrail."""

    def test_empty_weights(self):
        """Test with empty weights dictionaries."""
        trail = WeightAuditTrail(symbol='TEST')
        trail.capture(
            step_number=1,
            step_name='empty',
            weights_before={},
            weights_after={},
            adjustments=[]
        )
        summary = trail.get_summary()
        assert summary.initial_weights == {}
        assert summary.final_weights == {}

    def test_single_model(self):
        """Test with single model only."""
        trail = WeightAuditTrail(symbol='TEST')
        trail.capture(
            step_number=1,
            step_name='single',
            weights_before={},
            weights_after={'dcf': 100},
            adjustments=[]
        )
        summary = trail.get_summary()
        assert summary.final_weights == {'dcf': 100}

    def test_many_adjustments(self):
        """Test with many adjustments in one step."""
        trail = WeightAuditTrail(symbol='TEST')
        adjustments = [
            WeightAdjustment(f'model_{i}', 'source', 1.0 + i * 0.1)
            for i in range(20)
        ]
        trail.capture(
            step_number=1,
            step_name='many',
            weights_before={},
            weights_after={'dcf': 100},
            adjustments=adjustments
        )
        summary = trail.get_summary()
        assert summary.total_adjustments == 20

    def test_delta_pct_zero_before(self):
        """Test delta percentage when before is zero."""
        trail = WeightAuditTrail(symbol='TEST')
        trail.capture(
            step_number=1,
            step_name='from_zero',
            weights_before={'dcf': 0},
            weights_after={'dcf': 50},
            adjustments=[]
        )
        step = trail.steps[0]
        changes = step.get_changes()
        # Should handle zero before gracefully
        assert 'dcf' in changes
        assert changes['dcf']['delta_pct'] == 0  # Returns 0 when before is 0

    def test_started_at_timestamp(self):
        """Test that started_at is set."""
        trail = WeightAuditTrail(symbol='TEST')
        assert trail.started_at is not None
        # Should be valid ISO format
        from datetime import datetime
        datetime.fromisoformat(trail.started_at)  # Should not raise
