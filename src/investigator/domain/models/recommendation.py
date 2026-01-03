"""
Investment Recommendation Domain Model

Represents the final synthesis of fundamental and technical analysis
into actionable investment recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class InvestmentRecommendation:
    """
    Comprehensive investment recommendation synthesizing multiple analysis dimensions.

    This is the primary output of the synthesis process, combining:
    - Fundamental analysis scores
    - Technical analysis scores
    - Risk assessment
    - Valuation metrics
    - Investment thesis and strategy
    """

    # Identity
    symbol: str
    analysis_timestamp: datetime

    # Core Scores (0-100 scale)
    overall_score: float
    fundamental_score: float
    technical_score: float

    # Financial Statement Scores
    income_score: float  # Income statement quality
    cashflow_score: float  # Cash flow statement quality
    balance_score: float  # Balance sheet strength

    # Investment Dimension Scores
    growth_score: float  # Growth prospects
    value_score: float  # Value investment score
    business_quality_score: float  # Business quality from SEC analysis

    # Recommendation
    recommendation: str  # BUY, HOLD, SELL
    confidence: str  # HIGH, MEDIUM, LOW

    # Price & Targets
    current_price: Optional[float]
    price_target: Optional[float]
    stop_loss: Optional[float]

    # Investment Strategy
    investment_thesis: str
    time_horizon: str  # SHORT-TERM, MEDIUM-TERM, LONG-TERM
    position_size: str  # LARGE, MODERATE, SMALL, AVOID
    entry_strategy: str
    exit_strategy: str

    # Key Factors
    key_catalysts: List[str]
    key_risks: List[str]
    key_insights: List[str]

    # Data Quality
    data_quality_score: float
    data_quality_detailed: Optional[Dict[str, Any]] = None

    # Analysis Details (Optional)
    analysis_thinking: Optional[str] = None
    synthesis_details: Optional[str] = None

    # Historical & Comparative Data
    quarterly_metrics: Optional[List[Dict[str, Any]]] = None
    quarterly_trends: Optional[Dict[str, Any]] = None
    score_history: Optional[List[Dict[str, Any]]] = None
    score_trend: Optional[Dict[str, Any]] = None

    # Peer Analysis
    peer_valuation: Optional[Dict[str, Any]] = None
    competitive_positioning: Optional[Dict[str, Any]] = None
    peer_leaderboard: Optional[Dict[str, Any]] = None

    # Risk Analysis
    red_flags: Optional[List[Dict[str, Any]]] = None
    risk_scores: Optional[Dict[str, Any]] = None
    recession_performance: Optional[Dict[str, Any]] = None

    # Technical Analysis Details
    support_resistance: Optional[Dict[str, Any]] = None
    volume_profile: Optional[Dict[str, Any]] = None
    chart_patterns: Optional[Dict[str, Any]] = None

    # Advanced Analysis (Tier 3+)
    multi_year_trends: Optional[Dict[str, Any]] = None
    dcf_valuation: Optional[Dict[str, Any]] = None
    insider_trading: Optional[Dict[str, Any]] = None
    news_sentiment: Optional[Dict[str, Any]] = None

    # Simulation Results (Tier 4)
    monte_carlo_results: Optional[object] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "overall_score": self.overall_score,
            "fundamental_score": self.fundamental_score,
            "technical_score": self.technical_score,
            "income_score": self.income_score,
            "cashflow_score": self.cashflow_score,
            "balance_score": self.balance_score,
            "growth_score": self.growth_score,
            "value_score": self.value_score,
            "business_quality_score": self.business_quality_score,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "current_price": self.current_price,
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "investment_thesis": self.investment_thesis,
            "time_horizon": self.time_horizon,
            "position_size": self.position_size,
            "entry_strategy": self.entry_strategy,
            "exit_strategy": self.exit_strategy,
            "key_catalysts": self.key_catalysts,
            "key_risks": self.key_risks,
            "key_insights": self.key_insights,
            "data_quality_score": self.data_quality_score,
            "data_quality_detailed": self.data_quality_detailed,
            "analysis_thinking": self.analysis_thinking,
            "synthesis_details": self.synthesis_details,
            "quarterly_metrics": self.quarterly_metrics,
            "quarterly_trends": self.quarterly_trends,
            "score_history": self.score_history,
            "score_trend": self.score_trend,
            "peer_valuation": self.peer_valuation,
            "competitive_positioning": self.competitive_positioning,
            "peer_leaderboard": self.peer_leaderboard,
            "red_flags": self.red_flags,
            "risk_scores": self.risk_scores,
            "recession_performance": self.recession_performance,
            "support_resistance": self.support_resistance,
            "volume_profile": self.volume_profile,
            "chart_patterns": self.chart_patterns,
            "multi_year_trends": self.multi_year_trends,
            "dcf_valuation": self.dcf_valuation,
            "insider_trading": self.insider_trading,
            "news_sentiment": self.news_sentiment,
        }

    def get_risk_level(self) -> str:
        """Determine overall risk level based on scores and flags."""
        if self.red_flags and len(self.red_flags) > 5:
            return "HIGH"
        elif self.overall_score >= 70:
            return "LOW"
        elif self.overall_score >= 50:
            return "MEDIUM"
        else:
            return "HIGH"

    def is_buy_candidate(self) -> bool:
        """Determine if this is a buy candidate based on recommendation."""
        return self.recommendation == "BUY" and self.confidence in ["HIGH", "MEDIUM"]

    def get_summary(self) -> str:
        """Get a brief text summary of the recommendation."""
        return (
            f"{self.symbol}: {self.recommendation} ({self.confidence} confidence) | "
            f"Score: {self.overall_score:.1f}/100 | "
            f"Price: ${self.current_price:.2f}"
            if self.current_price
            else f"Score: {self.overall_score:.1f}/100"
        )


__all__ = ["InvestmentRecommendation"]
