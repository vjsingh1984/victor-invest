"""
Unit tests for RL models and dataclasses.
"""

import pytest
from datetime import date
from investigator.domain.services.rl.models import (
    ValuationContext,
    Experience,
    RewardSignal,
    TrainingMetrics,
    EvaluationMetrics,
    ABTestResults,
    GrowthStage,
    CompanySize,
    ABTestGroup,
)


class TestValuationContext:
    """Tests for ValuationContext dataclass."""

    def test_create_minimal_context(self):
        """Test creating context with minimal fields."""
        context = ValuationContext(
            symbol="AAPL",
            analysis_date=date(2024, 1, 15),
            sector="Technology",
            industry="Consumer Electronics",
        )
        assert context.symbol == "AAPL"
        assert context.sector == "Technology"
        assert context.industry == "Consumer Electronics"
        # Check that defaults are set
        assert context.growth_stage == GrowthStage.MATURE
        assert isinstance(context.company_size, CompanySize)

    def test_create_full_context(self):
        """Test creating context with all fields."""
        context = ValuationContext(
            symbol="ZS",
            analysis_date=date(2024, 1, 15),
            sector="Technology",
            industry="Software - Infrastructure",
            growth_stage=GrowthStage.HIGH_GROWTH,
            company_size=CompanySize.MID_CAP,
            profitability_score=0.75,
            pe_level=0.8,
            revenue_growth=0.35,
            fcf_margin=0.30,
            rule_of_40_score=65.0,
            payout_ratio=0.0,
            debt_to_equity=0.1,
            data_quality_score=85.0,
            quarters_available=8,
            technical_trend=0.5,
            market_sentiment=0.2,
            volatility=0.3,
            dcf_applicable=True,
            ggm_applicable=False,
            pe_applicable=True,
            ps_applicable=True,
            pb_applicable=True,
            evebitda_applicable=True,
        )
        assert context.symbol == "ZS"
        assert context.growth_stage == GrowthStage.HIGH_GROWTH
        assert context.rule_of_40_score == 65.0
        assert context.dcf_applicable is True
        assert context.ggm_applicable is False


class TestRewardSignal:
    """Tests for RewardSignal dataclass."""

    def test_create_reward_with_all_horizons(self):
        """Test creating reward with all time horizons."""
        reward = RewardSignal(
            reward_30d=0.5,
            reward_90d=0.8,
            reward_365d=0.9,
            direction_correct_90d=True,
        )
        assert reward.reward_90d == 0.8
        assert reward.direction_correct_90d is True

    def test_create_partial_reward(self):
        """Test creating reward with only some horizons."""
        reward = RewardSignal(
            reward_30d=0.6,
        )
        assert reward.reward_30d == 0.6
        assert reward.reward_90d is None

    def test_primary_reward_property(self):
        """Test primary_reward computed property."""
        reward = RewardSignal(
            reward_90d=0.8,
            reward_30d=0.5,
        )
        # primary_reward should return reward_90d if available
        assert reward.primary_reward == 0.8


class TestExperience:
    """Tests for Experience dataclass."""

    def test_create_experience(self):
        """Test creating an experience tuple."""
        context = ValuationContext(
            symbol="MSFT",
            analysis_date=date(2024, 1, 15),
            sector="Technology",
            industry="Software - Infrastructure",
        )
        reward = RewardSignal(reward_90d=0.65)

        experience = Experience(
            id=1,  # Required field
            symbol="MSFT",
            analysis_date=date(2024, 1, 15),
            context=context,
            weights_used={"dcf": 40, "pe": 30, "ps": 20, "ev_ebitda": 10},
            blended_fair_value=350.0,
            current_price=375.0,
            reward=reward,
            tier_classification="high_growth_strong",
        )
        assert experience.symbol == "MSFT"
        assert experience.blended_fair_value == 350.0
        assert experience.weights_used["dcf"] == 40
        assert experience.reward.reward_90d == 0.65
        assert experience.is_complete is True  # Has reward_90d


class TestGrowthStage:
    """Tests for GrowthStage enum."""

    def test_growth_stage_values(self):
        """Test all growth stage values exist."""
        assert GrowthStage.PRE_PROFIT.value == "pre_profit"
        assert GrowthStage.EARLY_GROWTH.value == "early_growth"
        assert GrowthStage.HIGH_GROWTH.value == "high_growth"
        assert GrowthStage.TRANSITIONING.value == "transitioning"
        assert GrowthStage.MATURE.value == "mature"
        assert GrowthStage.DIVIDEND_PAYING.value == "dividend_paying"


class TestCompanySize:
    """Tests for CompanySize enum."""

    def test_company_size_values(self):
        """Test all company size values exist."""
        assert CompanySize.MICRO_CAP.value == "micro_cap"
        assert CompanySize.SMALL_CAP.value == "small_cap"
        assert CompanySize.MID_CAP.value == "mid_cap"
        assert CompanySize.LARGE_CAP.value == "large_cap"
        assert CompanySize.MEGA_CAP.value == "mega_cap"


class TestABTestGroup:
    """Tests for ABTestGroup enum."""

    def test_ab_test_group_values(self):
        """Test A/B test group values."""
        assert ABTestGroup.RL.value == "rl"
        assert ABTestGroup.BASELINE.value == "baseline"


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_create_training_metrics(self):
        """Test creating training metrics."""
        from datetime import datetime

        metrics = TrainingMetrics(
            batch_id=1,
            batch_date=datetime(2024, 1, 15),
            policy_type="contextual_bandit",
            num_experiences=500,
            train_size=350,
            validation_size=75,
            test_size=75,
            train_loss=0.25,
            validation_loss=0.28,
            train_reward_mean=0.65,
            validation_reward_mean=0.62,
            epochs_completed=10,
            early_stopped=False,
            best_epoch=8,
        )
        assert metrics.batch_id == 1
        assert metrics.num_experiences == 500
        assert metrics.train_reward_mean == 0.65


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_create_evaluation_metrics(self):
        """Test creating evaluation metrics."""
        from datetime import datetime

        metrics = EvaluationMetrics(
            policy_type="hybrid",
            evaluation_date=datetime(2024, 3, 15),
            num_samples=100,
            mape=12.5,
            direction_accuracy=0.72,
            mean_reward=0.68,
            median_reward=0.70,
            std_reward=0.15,
            sector_performance={"Technology": {"mean_reward": 0.75, "count": 40}},
        )
        assert metrics.policy_type == "hybrid"
        assert metrics.mape == 12.5
        assert metrics.direction_accuracy == 0.72


class TestABTestResults:
    """Tests for ABTestResults dataclass."""

    def test_create_ab_test_results(self):
        """Test creating A/B test results."""
        results = ABTestResults(
            test_start_date=date(2024, 1, 1),
            test_end_date=date(2024, 3, 31),
            num_rl_samples=100,
            num_baseline_samples=400,
            rl_mean_reward=0.72,
            baseline_mean_reward=0.65,
            rl_mape=11.5,
            baseline_mape=14.2,
            rl_direction_accuracy=0.75,
            baseline_direction_accuracy=0.68,
            reward_p_value=0.01,
            mape_p_value=0.02,
            direction_p_value=0.03,
            reward_effect_size=0.6,
            mape_effect_size=0.5,
        )
        assert results.num_rl_samples == 100
        assert results.rl_mean_reward == 0.72
        assert results.is_significant is True  # p < 0.05 and rl > baseline
        assert results.recommendation == "FULL_ROLLOUT"  # Large effect size
