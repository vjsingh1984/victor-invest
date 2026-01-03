# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Escape hatches for Investment YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.

Example YAML usage:
    - id: check_data_quality
      type: condition
      condition: "data_quality_check"  # References escape hatch
      branches:
        "high": synthesize
        "acceptable": synthesize
        "low": request_review
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# =============================================================================
# Condition Functions
# =============================================================================


def data_quality_check(ctx: Dict[str, Any]) -> str:
    """Check if data quality is sufficient for synthesis.

    Evaluates data completeness, freshness, and consistency.

    Args:
        ctx: Workflow context with keys:
            - sec_data (dict): SEC filing data
            - market_data (dict): Market/price data
            - data_quality_score (float): Quality metric (0-100)

    Returns:
        "high", "acceptable", or "low"
    """
    sec_data = ctx.get("sec_data", {})
    market_data = ctx.get("market_data", {})
    quality_score = ctx.get("data_quality_score", 50)

    # Check data availability
    has_sec = bool(sec_data) and sec_data.get("status") == "success"
    has_market = bool(market_data) and market_data.get("status") == "success"

    if quality_score >= 80 and has_sec and has_market:
        return "high"

    if quality_score >= 60 or (has_sec and has_market):
        return "acceptable"

    return "low"


def valuation_confidence_check(ctx: Dict[str, Any]) -> str:
    """Assess confidence in valuation results.

    Multi-factor assessment based on model agreement and data quality.

    Args:
        ctx: Workflow context with keys:
            - valuation_models (dict): Results from valuation models
            - model_agreement_score (float): Agreement metric (0-100)
            - data_freshness (int): Days since last filing

    Returns:
        "high_confidence", "moderate_confidence", or "low_confidence"
    """
    models = ctx.get("valuation_models", {})
    agreement_score = ctx.get("model_agreement_score", 50)
    freshness = ctx.get("data_freshness", 365)

    model_count = len([m for m in models.values() if m])

    # High confidence: good agreement, multiple models, fresh data
    if agreement_score >= 80 and model_count >= 3 and freshness <= 90:
        return "high_confidence"

    if agreement_score >= 60 or model_count >= 2:
        return "moderate_confidence"

    return "low_confidence"


def recommendation_strength(ctx: Dict[str, Any]) -> str:
    """Determine recommendation strength from analysis.

    Args:
        ctx: Workflow context with keys:
            - composite_score (float): Overall score (0-100)
            - fundamental_score (float): Fundamental analysis score
            - technical_score (float): Technical analysis score

    Returns:
        "strong_buy", "buy", "hold", "sell", or "strong_sell"
    """
    composite = ctx.get("composite_score", 50)
    fundamental = ctx.get("fundamental_score", 50)
    technical = ctx.get("technical_score", 50)

    # Weight composite more heavily
    weighted_score = composite * 0.5 + fundamental * 0.3 + technical * 0.2

    if weighted_score >= 80:
        return "strong_buy"
    elif weighted_score >= 65:
        return "buy"
    elif weighted_score >= 35:
        return "hold"
    elif weighted_score >= 20:
        return "sell"
    else:
        return "strong_sell"


def should_request_peer_comparison(ctx: Dict[str, Any]) -> str:
    """Determine if peer comparison is needed.

    Args:
        ctx: Workflow context with keys:
            - has_peers (bool): Whether peers are available
            - sector (str): Company sector
            - analysis_mode (str): Current analysis mode

    Returns:
        "compare_peers" or "skip_peers"
    """
    has_peers = ctx.get("has_peers", False)
    mode = ctx.get("analysis_mode", "standard")

    if mode == "comprehensive" and has_peers:
        return "compare_peers"

    return "skip_peers"


def technical_signal_strength(ctx: Dict[str, Any]) -> str:
    """Assess technical signal strength.

    Args:
        ctx: Workflow context with keys:
            - rsi (float): RSI value
            - macd_signal (str): MACD signal (bullish/bearish/neutral)
            - trend (str): Price trend direction

    Returns:
        "strong_bullish", "bullish", "neutral", "bearish", or "strong_bearish"
    """
    rsi = ctx.get("rsi", 50)
    macd = ctx.get("macd_signal", "neutral")
    trend = ctx.get("trend", "neutral")

    bullish_signals = 0
    bearish_signals = 0

    # RSI
    if rsi < 30:
        bullish_signals += 2  # Oversold
    elif rsi < 40:
        bullish_signals += 1
    elif rsi > 70:
        bearish_signals += 2  # Overbought
    elif rsi > 60:
        bearish_signals += 1

    # MACD
    if macd == "bullish":
        bullish_signals += 1
    elif macd == "bearish":
        bearish_signals += 1

    # Trend
    if trend == "uptrend":
        bullish_signals += 1
    elif trend == "downtrend":
        bearish_signals += 1

    net_signal = bullish_signals - bearish_signals

    if net_signal >= 3:
        return "strong_bullish"
    elif net_signal >= 1:
        return "bullish"
    elif net_signal <= -3:
        return "strong_bearish"
    elif net_signal <= -1:
        return "bearish"
    else:
        return "neutral"


def insider_sentiment_check(ctx: Dict[str, Any]) -> str:
    """Assess insider trading sentiment.

    Args:
        ctx: Workflow context with keys:
            - insider_buys (int): Count of insider buys
            - insider_sells (int): Count of insider sells
            - buy_value (float): Total value of buys
            - sell_value (float): Total value of sells

    Returns:
        "bullish", "neutral", or "bearish"
    """
    buys = ctx.get("insider_buys", 0)
    sells = ctx.get("insider_sells", 0)
    buy_value = ctx.get("buy_value", 0)
    sell_value = ctx.get("sell_value", 0)

    # Count-based assessment
    if buys > sells * 2:
        return "bullish"
    if sells > buys * 2:
        return "bearish"

    # Value-based assessment
    if buy_value > sell_value * 1.5:
        return "bullish"
    if sell_value > buy_value * 1.5:
        return "bearish"

    return "neutral"


def risk_level_assessment(ctx: Dict[str, Any]) -> str:
    """Assess overall risk level.

    Args:
        ctx: Workflow context with keys:
            - beta (float): Stock beta
            - volatility (float): Historical volatility
            - debt_to_equity (float): D/E ratio
            - sector_risk (str): Sector risk level

    Returns:
        "low", "moderate", "high", or "very_high"
    """
    beta = ctx.get("beta", 1.0)
    volatility = ctx.get("volatility", 0.2)
    debt_ratio = ctx.get("debt_to_equity", 0.5)
    sector_risk = ctx.get("sector_risk", "moderate")

    risk_score = 0

    # Beta risk
    if beta > 1.5:
        risk_score += 2
    elif beta > 1.2:
        risk_score += 1

    # Volatility risk
    if volatility > 0.4:
        risk_score += 2
    elif volatility > 0.25:
        risk_score += 1

    # Leverage risk
    if debt_ratio > 2.0:
        risk_score += 2
    elif debt_ratio > 1.0:
        risk_score += 1

    # Sector risk
    if sector_risk == "high":
        risk_score += 1

    if risk_score >= 5:
        return "very_high"
    elif risk_score >= 3:
        return "high"
    elif risk_score >= 1:
        return "moderate"
    else:
        return "low"


# =============================================================================
# Transform Functions
# =============================================================================


def merge_analysis_results(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge results from parallel analysis operations.

    Args:
        ctx: Workflow context with parallel analysis results

    Returns:
        Merged and structured results
    """
    fundamental = ctx.get("fundamental_analysis", {})
    technical = ctx.get("technical_analysis", {})
    market_context = ctx.get("market_context", {})
    sec_insights = ctx.get("sec_insights", {})

    return {
        "merged_analysis": {
            "fundamental": fundamental,
            "technical": technical,
            "market_context": market_context,
            "sec_insights": sec_insights,
        },
        "analysis_complete": True,
        "component_count": sum([
            bool(fundamental),
            bool(technical),
            bool(market_context),
            bool(sec_insights),
        ]),
    }


def calculate_composite_score(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate weighted composite score from component analyses.

    Args:
        ctx: Workflow context with individual scores

    Returns:
        Composite score and component breakdown
    """
    fundamental = ctx.get("fundamental_score", 50)
    technical = ctx.get("technical_score", 50)
    sec = ctx.get("sec_score", 50)
    sentiment = ctx.get("sentiment_score", 50)

    # Default weights
    weights = ctx.get("weights", {
        "fundamental": 0.40,
        "technical": 0.25,
        "sec": 0.20,
        "sentiment": 0.15,
    })

    composite = (
        fundamental * weights.get("fundamental", 0.40) +
        technical * weights.get("technical", 0.25) +
        sec * weights.get("sec", 0.20) +
        sentiment * weights.get("sentiment", 0.15)
    )

    return {
        "composite_score": composite,
        "component_scores": {
            "fundamental": fundamental,
            "technical": technical,
            "sec": sec,
            "sentiment": sentiment,
        },
        "weights_applied": weights,
    }


def format_investment_thesis(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Format investment thesis from analysis.

    Args:
        ctx: Workflow context with analysis results

    Returns:
        Structured investment thesis
    """
    recommendation = ctx.get("recommendation", "hold")
    composite_score = ctx.get("composite_score", 50)
    catalysts = ctx.get("catalysts", [])
    risks = ctx.get("risks", [])

    thesis_text = f"Based on comprehensive analysis (score: {composite_score:.1f}/100), "
    thesis_text += f"the recommendation is {recommendation.upper()}. "

    if catalysts:
        thesis_text += f"Key catalysts: {', '.join(catalysts[:3])}. "
    if risks:
        thesis_text += f"Main risks: {', '.join(risks[:3])}."

    return {
        "investment_thesis": thesis_text,
        "recommendation": recommendation,
        "score": composite_score,
        "catalysts": catalysts[:5],
        "risks": risks[:5],
    }


def aggregate_peer_metrics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate metrics from peer comparison.

    Args:
        ctx: Workflow context with peer analyses

    Returns:
        Aggregated peer comparison metrics
    """
    target = ctx.get("target_analysis", {})
    peers = ctx.get("peer_analyses", [])

    if not peers:
        return {"peer_comparison": None, "status": "no_peers"}

    # Calculate peer averages
    peer_scores = [p.get("composite_score", 50) for p in peers if isinstance(p, dict)]

    if not peer_scores:
        return {"peer_comparison": None, "status": "invalid_peers"}

    avg_peer_score = sum(peer_scores) / len(peer_scores)
    target_score = target.get("composite_score", 50) if isinstance(target, dict) else 50

    relative_position = "above_average" if target_score > avg_peer_score else "below_average"

    return {
        "peer_comparison": {
            "target_score": target_score,
            "peer_average": avg_peer_score,
            "peer_count": len(peer_scores),
            "relative_position": relative_position,
            "percentile": sum(1 for s in peer_scores if target_score > s) / len(peer_scores) * 100,
        },
        "status": "success",
    }


# =============================================================================
# Registry Exports
# =============================================================================

# Conditions available in YAML workflows
CONDITIONS = {
    "data_quality_check": data_quality_check,
    "valuation_confidence_check": valuation_confidence_check,
    "recommendation_strength": recommendation_strength,
    "should_request_peer_comparison": should_request_peer_comparison,
    "technical_signal_strength": technical_signal_strength,
    "insider_sentiment_check": insider_sentiment_check,
    "risk_level_assessment": risk_level_assessment,
}

# Transforms available in YAML workflows
TRANSFORMS = {
    "merge_analysis_results": merge_analysis_results,
    "calculate_composite_score": calculate_composite_score,
    "format_investment_thesis": format_investment_thesis,
    "aggregate_peer_metrics": aggregate_peer_metrics,
}

__all__ = [
    # Conditions
    "data_quality_check",
    "valuation_confidence_check",
    "recommendation_strength",
    "should_request_peer_comparison",
    "technical_signal_strength",
    "insider_sentiment_check",
    "risk_level_assessment",
    # Transforms
    "merge_analysis_results",
    "calculate_composite_score",
    "format_investment_thesis",
    "aggregate_peer_metrics",
    # Registries
    "CONDITIONS",
    "TRANSFORMS",
]
