"""Recommendation and risk helpers extracted from InvestmentSynthesizer."""

from __future__ import annotations

from typing import Any, Dict, List


def calculate_consistency_bonus(quality_indicators: List[float]) -> float:
    """Calculate consistency bonus for quarterly quality indicators."""
    if len(quality_indicators) < 2:
        return 0.0

    mean_quality = sum(quality_indicators) / len(quality_indicators)
    variance = sum((x - mean_quality) ** 2 for x in quality_indicators) / len(quality_indicators)
    std_dev = variance**0.5

    max_bonus = 1.0
    return max(0.0, max_bonus - (std_dev / 2.0))


def determine_final_recommendation(overall_score: float, ai_recommendation: Dict[str, Any], data_quality: float) -> Dict[str, str]:
    """Determine final recommendation with score and data-quality adjustments."""
    if "investment_recommendation" in ai_recommendation:
        inv_rec = ai_recommendation["investment_recommendation"]
        base_recommendation = inv_rec.get("recommendation", "HOLD")
        confidence = inv_rec.get("confidence_level", "MEDIUM")
    else:
        rec_data = ai_recommendation.get("recommendation", "HOLD")
        if isinstance(rec_data, dict):
            base_recommendation = rec_data.get("rating", "HOLD")
            confidence = rec_data.get("confidence", "LOW")
        else:
            base_recommendation = rec_data if isinstance(rec_data, str) else "HOLD"
            confidence = ai_recommendation.get("confidence", "MEDIUM")

    if data_quality < 0.5:
        confidence = "LOW"
        if base_recommendation in ["STRONG BUY", "STRONG SELL"]:
            base_recommendation = base_recommendation.replace("STRONG ", "")

    if overall_score >= 8.0 and base_recommendation not in ["BUY", "STRONG BUY"]:
        base_recommendation = "BUY"
    elif overall_score <= 3.0 and base_recommendation not in ["SELL", "STRONG SELL"]:
        base_recommendation = "SELL"
    elif 4.0 <= overall_score <= 6.0 and base_recommendation in ["STRONG BUY", "STRONG SELL"]:
        base_recommendation = "HOLD"

    return {"recommendation": base_recommendation, "confidence": confidence}


def calculate_price_target(symbol: str, ai_recommendation: Dict[str, Any], current_price: float, logger: Any) -> float:
    """Calculate 12-month target price from structured fields or score mapping."""
    if "investment_recommendation" in ai_recommendation:
        target_data = ai_recommendation["investment_recommendation"].get("target_price", {})
        if target_data.get("12_month_target"):
            return target_data["12_month_target"]

    ai_targets = ai_recommendation.get("price_targets", {})
    if ai_targets.get("12_month"):
        return ai_targets["12_month"]

    if current_price <= 0:
        logger.warning(f"No current price available for {symbol}, using placeholder for target calculation")
        current_price = 100

    overall_score = 5.0
    if "composite_scores" in ai_recommendation:
        overall_score = ai_recommendation["composite_scores"].get("overall_score", 5.0)
    elif "overall_score" in ai_recommendation:
        overall_score = ai_recommendation.get("overall_score", 5.0)

    if overall_score >= 8.0:
        expected_return = 0.15
    elif overall_score >= 6.5:
        expected_return = 0.10
    elif overall_score >= 5.0:
        expected_return = 0.05
    else:
        expected_return = -0.05

    price_target = round(current_price * (1 + expected_return), 2)
    logger.info(
        f"Calculated price target for {symbol}: ${price_target:.2f} "
        f"(current: ${current_price:.2f}, score: {overall_score:.1f})"
    )
    return price_target


def calculate_stop_loss(current_price: float, recommendation: Dict[str, Any], overall_score: float) -> float:
    """Calculate stop loss level from recommendation and conviction."""
    if not current_price or current_price <= 0:
        return 0

    rec_type = recommendation.get("recommendation", "HOLD")
    if "STRONG BUY" in rec_type:
        stop_loss_pct = 0.12
    elif "BUY" in rec_type:
        stop_loss_pct = 0.10
    elif "HOLD" in rec_type:
        stop_loss_pct = 0.08
    else:
        stop_loss_pct = 0.05

    if overall_score < 4.0:
        stop_loss_pct *= 0.5

    return round(current_price * (1 - stop_loss_pct), 2)


def extract_position_size(ai_recommendation: Dict[str, Any]) -> str:
    """Extract normalized position size bucket."""
    if "investment_recommendation" in ai_recommendation:
        pos_sizing = ai_recommendation["investment_recommendation"].get("position_sizing", {})
        weight = pos_sizing.get("recommended_weight", 0.0)
        if weight >= 0.05:
            return "LARGE"
        if weight >= 0.03:
            return "MODERATE"
        if weight > 0:
            return "SMALL"
    return ai_recommendation.get("position_size", "MODERATE")


def extract_catalysts(ai_recommendation: Dict[str, Any]) -> List[str]:
    """Extract up to three catalysts from structured recommendation payloads."""
    catalysts: List[str] = []

    if "key_catalysts" in ai_recommendation:
        cat_data = ai_recommendation["key_catalysts"]
        if isinstance(cat_data, list):
            for cat in cat_data[:3]:
                if isinstance(cat, dict):
                    catalysts.append(cat.get("catalyst", ""))
                elif isinstance(cat, str):
                    catalysts.append(cat)

    return catalysts or ai_recommendation.get("catalysts", [])
