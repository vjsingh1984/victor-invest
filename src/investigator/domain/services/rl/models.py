"""
Data Models for RL-Based Valuation System

Defines core data structures for:
- ValuationContext: State representation for RL agent
- Experience: Training sample (state, action, reward)
- Metrics: Training and evaluation metrics
- Rewards: Reward signal calculation
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class GrowthStage(Enum):
    """Company growth stage classification."""

    PRE_PROFIT = "pre_profit"
    EARLY_GROWTH = "early_growth"
    HIGH_GROWTH = "high_growth"
    TRANSITIONING = "transitioning"
    MATURE = "mature"
    DIVIDEND_PAYING = "dividend_paying"


class CompanySize(Enum):
    """Market cap classification."""

    MICRO_CAP = "micro_cap"  # < $300M
    SMALL_CAP = "small_cap"  # $300M - $2B
    MID_CAP = "mid_cap"  # $2B - $10B
    LARGE_CAP = "large_cap"  # $10B - $200B
    MEGA_CAP = "mega_cap"  # > $200B


class ABTestGroup(Enum):
    """A/B test assignment groups."""

    RL = "rl"
    BASELINE = "baseline"
    CONTROL = "control"


class HoldingPeriod(Enum):
    """Recommended holding periods for positions."""

    ONE_MONTH = "1m"
    THREE_MONTHS = "3m"
    SIX_MONTHS = "6m"
    ONE_YEAR = "12m"
    EIGHTEEN_MONTHS = "18m"
    TWO_YEARS = "24m"
    THREE_YEARS = "36m"

    @property
    def days(self) -> int:
        """Convert to days."""
        mapping = {
            "1m": 30,
            "3m": 90,
            "6m": 180,
            "12m": 365,
            "18m": 540,
            "24m": 730,
            "36m": 1095,
        }
        return mapping[self.value]

    @classmethod
    def from_days(cls, days: int) -> "HoldingPeriod":
        """Get closest holding period for a given number of days."""
        periods = [
            (30, cls.ONE_MONTH),
            (90, cls.THREE_MONTHS),
            (180, cls.SIX_MONTHS),
            (365, cls.ONE_YEAR),
            (540, cls.EIGHTEEN_MONTHS),
            (730, cls.TWO_YEARS),
            (1095, cls.THREE_YEARS),
        ]
        for period_days, period in periods:
            if days <= period_days * 1.25:  # Allow 25% tolerance
                return period
        return cls.THREE_YEARS


@dataclass
class ValuationContext:
    """
    State representation for RL agent.

    Contains all features the RL agent observes when making
    weight predictions. Features are normalized for training.

    Attributes:
        symbol: Stock ticker
        analysis_date: Date of analysis

        # Company Classification (categorical)
        sector: GICS sector (11 categories)
        industry: GICS industry (~70 categories)
        growth_stage: From GrowthStage enum
        company_size: From CompanySize enum

        # Fundamental Metrics (continuous, normalized 0-1)
        profitability_score: From ProfitabilityClassifier
        pe_level: Normalized P/E (0=deep value, 1=extreme growth)
        revenue_growth: Revenue growth rate
        fcf_margin: Free cash flow margin
        rule_of_40_score: Revenue growth + FCF margin (SaaS metric)
        payout_ratio: Dividend payout ratio
        debt_to_equity: Leverage ratio

        # Data Quality
        data_quality_score: From DataQualityScorer (0-100)
        quarters_available: Number of quarterly data points

        # Market Context
        technical_trend: -1 (bearish) to +1 (bullish)
        market_sentiment: -1 to +1
        volatility: Normalized volatility (0-1)

        # Model Applicability Flags
        dcf_applicable: Whether DCF model can be used
        ggm_applicable: Whether Gordon Growth Model can be used
        pe_applicable: Whether P/E model can be used
        ps_applicable: Whether P/S model can be used
        pb_applicable: Whether P/B model can be used
        evebitda_applicable: Whether EV/EBITDA model can be used
    """

    # Identification
    symbol: str
    analysis_date: date

    # Company Classification (categorical)
    sector: str = "Unknown"
    industry: str = "Unknown"
    growth_stage: GrowthStage = GrowthStage.MATURE
    company_size: CompanySize = CompanySize.MID_CAP

    # Fundamental Metrics (continuous)
    profitability_score: float = 0.5
    pe_level: float = 0.5
    revenue_growth: float = 0.0
    fcf_margin: float = 0.0
    rule_of_40_score: float = 0.0
    payout_ratio: float = 0.0
    debt_to_equity: float = 0.0
    gross_margin: float = 0.0
    operating_margin: float = 0.0

    # Data Quality
    data_quality_score: float = 50.0
    quarters_available: int = 0

    # Market Context
    technical_trend: float = 0.0
    market_sentiment: float = 0.0
    volatility: float = 0.5

    # Technical Indicators (normalized)
    rsi_14: float = 50.0  # 0-100, neutral at 50
    macd_histogram: float = 0.0  # Normalized MACD histogram
    obv_trend: float = 0.0  # -1 (bearish) to +1 (bullish)
    adx_14: float = 25.0  # 0-100, trend strength
    stoch_k: float = 50.0  # 0-100
    mfi_14: float = 50.0  # 0-100, money flow

    # Entry/Exit Signal Features
    entry_signal_strength: float = 0.0  # -1 (avoid) to +1 (strong buy)
    exit_signal_strength: float = 0.0  # -1 (hold) to +1 (strong sell)
    signal_confluence: float = 0.0  # How many signals agree (-1 to +1)
    days_from_support: float = 0.5  # 0 (at support) to 1 (at resistance)
    risk_reward_ratio: float = 2.0  # Expected R/R from entry

    # Insider Sentiment Features (from Form 4 filings)
    insider_sentiment: float = 0.0  # -1 (heavy selling) to +1 (heavy buying)
    insider_buy_ratio: float = 0.5  # 0 (all sells) to 1 (all buys)
    insider_transaction_value: float = 0.0  # Normalized net transaction value
    insider_cluster_signal: float = 0.0  # 1 if cluster detected, 0 otherwise
    insider_key_exec_activity: float = 0.0  # Key executive (CEO/CFO/Director) activity

    # Economic Indicators (from Regional Fed and CBOE)
    gdpnow: float = 2.0  # Atlanta Fed GDPNow estimate (typical range: -5 to 10)
    cfnai: float = 0.0  # Chicago Fed National Activity Index (-1 to +1 typical)
    nfci: float = 0.0  # National Financial Conditions Index (0 = avg, + = tighter)
    kcfsi: float = 0.0  # Kansas City Financial Stress Index (0 = avg, + = stress)
    inflation_expectations: float = 2.5  # Cleveland Fed 1-year inflation expectation
    recession_probability: float = 0.15  # NY Fed recession probability (0-1)
    empire_state_mfg: float = 0.0  # NY Fed Empire State Manufacturing (-50 to +50)
    # CBOE Volatility Data
    vix: float = 18.0  # VIX level (typical: 12-35)
    vix_term_structure: float = 1.0  # VIX3M/VIX ratio (>1 = contango, <1 = backwardation)
    skew: float = 120.0  # CBOE SKEW index (typical: 100-150, >130 = elevated)
    volatility_regime: int = 2  # 0=very_low, 1=low, 2=normal, 3=elevated, 4=high, 5=extreme
    is_backwardation: bool = False  # VIX term structure inverted (fear signal)

    # Valuation Gap Features (for position filtering)
    valuation_gap: float = 0.0  # (FV - Price) / Price, negative = "overvalued"
    valuation_confidence: float = 0.5  # 0 = low confidence, 1 = high confidence
    position_signal: int = 0  # 1 = Long, -1 = Short, 0 = Skip/No position

    # Optimal Holding Period (learned from backtest outcomes)
    optimal_holding_period: Optional[str] = None  # "1m", "3m", "6m", "12m", "18m", "24m", "36m"
    optimal_holding_reward: float = 0.0  # Best reward achieved at optimal period

    # Model Applicability Flags
    dcf_applicable: bool = True
    ggm_applicable: bool = False
    pe_applicable: bool = True
    ps_applicable: bool = True
    pb_applicable: bool = True
    evebitda_applicable: bool = True

    # Additional context
    fiscal_period: Optional[str] = None
    current_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "analysis_date": self.analysis_date.isoformat(),
            "sector": self.sector,
            "industry": self.industry,
            "growth_stage": self.growth_stage.value,
            "company_size": self.company_size.value,
            "profitability_score": self.profitability_score,
            "pe_level": self.pe_level,
            "revenue_growth": self.revenue_growth,
            "fcf_margin": self.fcf_margin,
            "rule_of_40_score": self.rule_of_40_score,
            "payout_ratio": self.payout_ratio,
            "debt_to_equity": self.debt_to_equity,
            "gross_margin": self.gross_margin,
            "operating_margin": self.operating_margin,
            "data_quality_score": self.data_quality_score,
            "quarters_available": self.quarters_available,
            "technical_trend": self.technical_trend,
            "market_sentiment": self.market_sentiment,
            "volatility": self.volatility,
            # Technical indicators
            "rsi_14": self.rsi_14,
            "macd_histogram": self.macd_histogram,
            "obv_trend": self.obv_trend,
            "adx_14": self.adx_14,
            "stoch_k": self.stoch_k,
            "mfi_14": self.mfi_14,
            # Entry/Exit signals
            "entry_signal_strength": self.entry_signal_strength,
            "exit_signal_strength": self.exit_signal_strength,
            "signal_confluence": self.signal_confluence,
            "days_from_support": self.days_from_support,
            "risk_reward_ratio": self.risk_reward_ratio,
            # Insider sentiment features
            "insider_sentiment": self.insider_sentiment,
            "insider_buy_ratio": self.insider_buy_ratio,
            "insider_transaction_value": self.insider_transaction_value,
            "insider_cluster_signal": self.insider_cluster_signal,
            "insider_key_exec_activity": self.insider_key_exec_activity,
            # Economic indicators (Regional Fed)
            "gdpnow": self.gdpnow,
            "cfnai": self.cfnai,
            "nfci": self.nfci,
            "kcfsi": self.kcfsi,
            "inflation_expectations": self.inflation_expectations,
            "recession_probability": self.recession_probability,
            "empire_state_mfg": self.empire_state_mfg,
            # CBOE volatility data
            "vix": self.vix,
            "vix_term_structure": self.vix_term_structure,
            "skew": self.skew,
            "volatility_regime": self.volatility_regime,
            "is_backwardation": self.is_backwardation,
            # Valuation gap features
            "valuation_gap": self.valuation_gap,
            "valuation_confidence": self.valuation_confidence,
            "position_signal": self.position_signal,
            # Optimal holding period
            "optimal_holding_period": self.optimal_holding_period,
            "optimal_holding_reward": self.optimal_holding_reward,
            "dcf_applicable": self.dcf_applicable,
            "ggm_applicable": self.ggm_applicable,
            "pe_applicable": self.pe_applicable,
            "ps_applicable": self.ps_applicable,
            "pb_applicable": self.pb_applicable,
            "evebitda_applicable": self.evebitda_applicable,
            "fiscal_period": self.fiscal_period,
            "current_price": self.current_price,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValuationContext":
        """Create from dictionary."""
        # Parse enums
        growth_stage = GrowthStage(data.get("growth_stage", "mature"))
        company_size = CompanySize(data.get("company_size", "mid_cap"))

        # Parse date
        analysis_date = data.get("analysis_date")
        if isinstance(analysis_date, str):
            analysis_date = date.fromisoformat(analysis_date)
        elif analysis_date is None:
            analysis_date = date.today()

        return cls(
            symbol=data.get("symbol", ""),
            analysis_date=analysis_date,
            sector=data.get("sector", "Unknown"),
            industry=data.get("industry", "Unknown"),
            growth_stage=growth_stage,
            company_size=company_size,
            profitability_score=data.get("profitability_score", 0.5),
            pe_level=data.get("pe_level", 0.5),
            revenue_growth=data.get("revenue_growth", 0.0),
            fcf_margin=data.get("fcf_margin", 0.0),
            rule_of_40_score=data.get("rule_of_40_score", 0.0),
            payout_ratio=data.get("payout_ratio", 0.0),
            debt_to_equity=data.get("debt_to_equity", 0.0),
            gross_margin=data.get("gross_margin", 0.0),
            operating_margin=data.get("operating_margin", 0.0),
            data_quality_score=data.get("data_quality_score", 50.0),
            quarters_available=data.get("quarters_available", 0),
            technical_trend=data.get("technical_trend", 0.0),
            market_sentiment=data.get("market_sentiment", 0.0),
            volatility=data.get("volatility", 0.5),
            # Technical indicators
            rsi_14=data.get("rsi_14", 50.0),
            macd_histogram=data.get("macd_histogram", 0.0),
            obv_trend=data.get("obv_trend", 0.0),
            adx_14=data.get("adx_14", 25.0),
            stoch_k=data.get("stoch_k", 50.0),
            mfi_14=data.get("mfi_14", 50.0),
            # Entry/Exit signals
            entry_signal_strength=data.get("entry_signal_strength", 0.0),
            exit_signal_strength=data.get("exit_signal_strength", 0.0),
            signal_confluence=data.get("signal_confluence", 0.0),
            days_from_support=data.get("days_from_support", 0.5),
            risk_reward_ratio=data.get("risk_reward_ratio", 2.0),
            # Insider sentiment features
            insider_sentiment=data.get("insider_sentiment", 0.0),
            insider_buy_ratio=data.get("insider_buy_ratio", 0.5),
            insider_transaction_value=data.get("insider_transaction_value", 0.0),
            insider_cluster_signal=data.get("insider_cluster_signal", 0.0),
            insider_key_exec_activity=data.get("insider_key_exec_activity", 0.0),
            # Economic indicators (Regional Fed)
            gdpnow=data.get("gdpnow", 2.0),
            cfnai=data.get("cfnai", 0.0),
            nfci=data.get("nfci", 0.0),
            kcfsi=data.get("kcfsi", 0.0),
            inflation_expectations=data.get("inflation_expectations", 2.5),
            recession_probability=data.get("recession_probability", 0.15),
            empire_state_mfg=data.get("empire_state_mfg", 0.0),
            # CBOE volatility data
            vix=data.get("vix", 18.0),
            vix_term_structure=data.get("vix_term_structure", 1.0),
            skew=data.get("skew", 120.0),
            volatility_regime=data.get("volatility_regime", 2),
            is_backwardation=data.get("is_backwardation", False),
            # Valuation gap features
            valuation_gap=data.get("valuation_gap", 0.0),
            valuation_confidence=data.get("valuation_confidence", 0.5),
            position_signal=data.get("position_signal", 0),
            # Optimal holding period
            optimal_holding_period=data.get("optimal_holding_period"),
            optimal_holding_reward=data.get("optimal_holding_reward", 0.0),
            dcf_applicable=data.get("dcf_applicable", True),
            ggm_applicable=data.get("ggm_applicable", False),
            pe_applicable=data.get("pe_applicable", True),
            ps_applicable=data.get("ps_applicable", True),
            pb_applicable=data.get("pb_applicable", True),
            evebitda_applicable=data.get("evebitda_applicable", True),
            fiscal_period=data.get("fiscal_period"),
            current_price=data.get("current_price"),
        )


@dataclass
class RewardSignal:
    """
    Reward signal calculated from prediction outcomes.

    Supports multiple holding periods for comprehensive reward analysis:
    - 1m, 3m, 6m (short-term: momentum/earnings plays)
    - 12m, 18m (medium-term: business cycle)
    - 24m, 36m (long-term: fundamental value thesis)

    Attributes:
        reward_30d: Reward based on 30-day outcome (-1 to 1)
        reward_90d: Reward based on 90-day outcome (-1 to 1)
        reward_365d: Reward based on 365-day outcome (-1 to 1)
        direction_correct_30d: Did we predict direction correctly at 30d?
        direction_correct_90d: Did we predict direction correctly at 90d?
        error_pct_30d: Absolute percentage error at 30d
        error_pct_90d: Absolute percentage error at 90d
        error_pct_365d: Absolute percentage error at 365d
        multi_period_rewards: Dict of rewards by holding period (1m, 3m, 6m, etc.)
        optimal_period: Best performing holding period
        optimal_reward: Reward at optimal holding period
    """

    reward_30d: Optional[float] = None
    reward_90d: Optional[float] = None
    reward_365d: Optional[float] = None
    direction_correct_30d: Optional[bool] = None
    direction_correct_90d: Optional[bool] = None
    error_pct_30d: Optional[float] = None
    error_pct_90d: Optional[float] = None
    error_pct_365d: Optional[float] = None

    # Multi-period rewards (new)
    multi_period_rewards: Dict[str, Optional[float]] = field(default_factory=dict)
    optimal_period: Optional[str] = None  # "1m", "3m", "6m", "12m", "18m", "24m", "36m"
    optimal_reward: Optional[float] = None

    @property
    def primary_reward(self) -> Optional[float]:
        """Primary reward signal (90-day weighted)."""
        if self.reward_90d is not None:
            return self.reward_90d
        return self.reward_30d

    def get_best_holding_period(self) -> Tuple[Optional[str], Optional[float]]:
        """Find the holding period with highest reward."""
        if not self.multi_period_rewards:
            return self.optimal_period, self.optimal_reward

        best_period = None
        best_reward = -float("inf")
        for period, reward in self.multi_period_rewards.items():
            if reward is not None and reward > best_reward:
                best_reward = reward
                best_period = period

        if best_period is None:
            return None, None
        return best_period, best_reward


@dataclass
class PerModelReward:
    """Reward signal for individual valuation model."""

    model_name: str
    fair_value: float
    actual_price_30d: Optional[float] = None
    actual_price_90d: Optional[float] = None
    reward_30d: Optional[float] = None
    reward_90d: Optional[float] = None
    error_pct_30d: Optional[float] = None
    error_pct_90d: Optional[float] = None


@dataclass
class Experience:
    """
    Training experience tuple (state, action, reward).

    Used for RL policy training.

    Attributes:
        id: Database record ID
        symbol: Stock ticker
        analysis_date: Date prediction was made
        context: ValuationContext (state)
        weights_used: Model weights that were used (action)
        tier_classification: Tier that was selected
        blended_fair_value: Predicted fair value
        current_price: Price at prediction time
        reward: Calculated reward signal
        per_model_rewards: Individual model accuracy
    """

    id: int
    symbol: str
    analysis_date: date
    context: ValuationContext
    weights_used: Dict[str, float]
    tier_classification: str
    blended_fair_value: float
    current_price: float
    reward: RewardSignal
    per_model_rewards: Optional[Dict[str, PerModelReward]] = None

    @property
    def is_complete(self) -> bool:
        """Check if experience has outcome data for training."""
        return self.reward.reward_90d is not None


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""

    batch_id: int
    batch_date: datetime
    policy_type: str

    # Data stats
    num_experiences: int
    train_size: int
    validation_size: int
    test_size: int

    # Training metrics
    train_loss: float
    validation_loss: float
    train_reward_mean: float
    validation_reward_mean: float

    # Improvement vs baseline
    baseline_mape: Optional[float] = None
    rl_mape: Optional[float] = None
    mape_improvement_pct: Optional[float] = None
    direction_accuracy: Optional[float] = None

    # Training metadata
    epochs_completed: int = 0
    early_stopped: bool = False
    best_epoch: int = 0


@dataclass
class EvaluationMetrics:
    """Metrics from policy evaluation."""

    policy_type: str
    evaluation_date: datetime
    num_samples: int

    # Accuracy metrics
    mape: float  # Mean Absolute Percentage Error
    direction_accuracy: float  # % correct buy/sell

    # Reward metrics
    mean_reward: float
    median_reward: float
    std_reward: float

    # Per-sector breakdown
    sector_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-model contribution
    model_contribution: Dict[str, float] = field(default_factory=dict)

    # Comparison to baseline
    baseline_mape: Optional[float] = None
    improvement_pct: Optional[float] = None


@dataclass
class ABTestResults:
    """Results from A/B testing RL policy vs baseline."""

    test_start_date: date
    test_end_date: date
    num_rl_samples: int
    num_baseline_samples: int

    # RL performance
    rl_mean_reward: float
    rl_mape: float
    rl_direction_accuracy: float

    # Baseline performance
    baseline_mean_reward: float
    baseline_mape: float
    baseline_direction_accuracy: float

    # Statistical significance
    reward_p_value: float
    mape_p_value: float
    direction_p_value: float

    # Effect sizes
    reward_effect_size: float  # Cohen's d
    mape_effect_size: float

    @property
    def is_significant(self) -> bool:
        """Check if RL improvement is statistically significant (p < 0.05)."""
        return self.reward_p_value < 0.05 and self.rl_mean_reward > self.baseline_mean_reward

    @property
    def recommendation(self) -> str:
        """Get recommendation based on test results."""
        if not self.is_significant:
            if self.num_rl_samples < 100:
                return "CONTINUE_TEST"  # Need more samples
            return "KEEP_BASELINE"  # RL not better

        if abs(self.reward_effect_size) < 0.2:
            return "CONTINUE_TEST"  # Effect too small
        elif abs(self.reward_effect_size) < 0.5:
            return "GRADUAL_ROLLOUT"  # Moderate effect
        else:
            return "FULL_ROLLOUT"  # Large effect


@dataclass
class PolicyCheckpoint:
    """Saved policy state for persistence and rollback."""

    policy_type: str
    version: str
    created_at: datetime
    model_path: str
    normalizer_path: str
    config_snapshot: Dict[str, Any]
    training_metrics: Optional[TrainingMetrics] = None
    evaluation_metrics: Optional[EvaluationMetrics] = None
    is_active: bool = False
