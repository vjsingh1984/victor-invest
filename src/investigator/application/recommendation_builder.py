"""
Recommendation Builder - Extracted from InvestmentSynthesizer for SRP.

This module handles building investment recommendations:
- Final recommendation determination
- Price target calculation
- Position size extraction
- Catalyst extraction

Part of Phase 5 refactoring to break up monolithic synthesizer.py.

Author: InvestiGator Team
Date: 2025-01-05
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RecommendationBuilder:
    """
    Builds investment recommendations from analysis data.

    Extracted from InvestmentSynthesizer to follow Single Responsibility Principle.
    All recommendation building logic is centralized here.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize recommendation builder.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def determine_final_recommendation(
        self,
        overall_score: float,
        ai_recommendation: Dict,
        data_quality: float,
    ) -> Dict:
        """
        Determine final recommendation with risk management.

        Logic:
        1. Extract base recommendation from AI response
        2. Adjust confidence for low data quality
        3. Downgrade STRONG recommendations if data quality < 0.5
        4. Adjust recommendation based on score thresholds

        Args:
            overall_score: Overall investment score (0-10)
            ai_recommendation: AI recommendation dictionary
            data_quality: Data quality score (0-1)

        Returns:
            Dict with 'recommendation' and 'confidence' keys
        """
        # Try to get recommendation from structured response first
        if "investment_recommendation" in ai_recommendation:
            inv_rec = ai_recommendation["investment_recommendation"]
            base_recommendation = inv_rec.get("recommendation", "HOLD")
            confidence = inv_rec.get("confidence_level", "MEDIUM")
        else:
            # Handle case where recommendation might be a dict due to JSON parsing errors
            rec_data = ai_recommendation.get("recommendation", "HOLD")
            if isinstance(rec_data, dict):
                base_recommendation = rec_data.get("rating", "HOLD")
                confidence = rec_data.get("confidence", "LOW")
            else:
                base_recommendation = rec_data if isinstance(rec_data, str) else "HOLD"
                confidence = ai_recommendation.get("confidence", "MEDIUM")

        # Adjust for data quality
        if data_quality < 0.5:
            confidence = "LOW"
            if base_recommendation in ["STRONG BUY", "STRONG SELL"]:
                base_recommendation = base_recommendation.replace("STRONG ", "")

        # Adjust based on score thresholds
        if overall_score >= 8.0 and base_recommendation not in ["BUY", "STRONG BUY"]:
            base_recommendation = "BUY"
        elif overall_score <= 3.0 and base_recommendation not in ["SELL", "STRONG SELL"]:
            base_recommendation = "SELL"
        elif 4.0 <= overall_score <= 6.0 and base_recommendation in ["STRONG BUY", "STRONG SELL"]:
            base_recommendation = "HOLD"

        return {"recommendation": base_recommendation, "confidence": confidence}

    def calculate_price_target(
        self,
        symbol: str,
        llm_responses: Dict,
        ai_recommendation: Dict,
        current_price: float,
    ) -> float:
        """
        Calculate sophisticated price target.

        Extraction order:
        1. Structured 12-month target from investment_recommendation
        2. Legacy format price_targets
        3. Calculate from overall score using expected return mapping

        Args:
            symbol: Stock symbol
            llm_responses: LLM analysis responses
            ai_recommendation: AI recommendation dictionary
            current_price: Current stock price

        Returns:
            Calculated price target
        """
        # Try to extract from structured AI recommendation first
        if "investment_recommendation" in ai_recommendation:
            target_data = ai_recommendation["investment_recommendation"].get("target_price", {})
            if target_data.get("12_month_target"):
                return target_data["12_month_target"]

        # Try legacy format
        ai_targets = ai_recommendation.get("price_targets", {})
        if ai_targets.get("12_month"):
            return ai_targets["12_month"]

        # Use current price passed in, fallback to reasonable default if price is 0
        if current_price <= 0:
            self.logger.warning(
                f"No current price available for {symbol}, using placeholder for target calculation"
            )
            current_price = 100  # Fallback only as last resort

        # Extract overall score from different possible locations
        overall_score = 5.0  # Default
        if "composite_scores" in ai_recommendation:
            overall_score = ai_recommendation["composite_scores"].get("overall_score", 5.0)
        elif "overall_score" in ai_recommendation:
            overall_score = ai_recommendation.get("overall_score", 5.0)

        # Expected return mapping based on score
        if overall_score >= 8.0:
            expected_return = 0.15  # 15% (more conservative for institutional)
        elif overall_score >= 6.5:
            expected_return = 0.10  # 10%
        elif overall_score >= 5.0:
            expected_return = 0.05  # 5%
        else:
            expected_return = -0.05  # -5%

        price_target = round(current_price * (1 + expected_return), 2)
        self.logger.info(
            f"Calculated price target for {symbol}: ${price_target:.2f} "
            f"(current: ${current_price:.2f}, score: {overall_score:.1f})"
        )

        return price_target

    def extract_position_size(self, ai_recommendation: Dict) -> str:
        """
        Extract position size recommendation.

        Position sizing based on recommended weight:
        - >= 5%: LARGE
        - >= 3%: MODERATE
        - > 0%: SMALL
        - Default: MODERATE

        Args:
            ai_recommendation: AI recommendation dictionary

        Returns:
            Position size string (LARGE, MODERATE, or SMALL)
        """
        if "investment_recommendation" in ai_recommendation:
            pos_sizing = ai_recommendation["investment_recommendation"].get("position_sizing", {})
            weight = pos_sizing.get("recommended_weight", 0.0)
            if weight >= 0.05:
                return "LARGE"
            elif weight >= 0.03:
                return "MODERATE"
            elif weight > 0:
                return "SMALL"
        return ai_recommendation.get("position_size", "MODERATE")

    def extract_catalysts(self, ai_recommendation: Dict) -> List[str]:
        """
        Extract key catalysts from recommendation.

        Supports multiple formats:
        - List of dicts with 'catalyst' key
        - List of strings
        - Fallback to 'catalysts' list

        Limited to 3 catalysts maximum.

        Args:
            ai_recommendation: AI recommendation dictionary

        Returns:
            List of catalyst strings (max 3)
        """
        catalysts = []

        # Try structured format first
        if "key_catalysts" in ai_recommendation:
            cat_data = ai_recommendation["key_catalysts"]
            if isinstance(cat_data, list):
                for cat in cat_data[:3]:
                    if isinstance(cat, dict):
                        catalysts.append(cat.get("catalyst", ""))
                    elif isinstance(cat, str):
                        catalysts.append(cat)

        # Fallback to simple list
        return catalysts or ai_recommendation.get("catalysts", [])


# Singleton instance
_builder_instance: Optional[RecommendationBuilder] = None


def get_recommendation_builder(
    logger: Optional[logging.Logger] = None,
) -> RecommendationBuilder:
    """
    Get singleton RecommendationBuilder instance.

    Args:
        logger: Optional logger (only used on first call)

    Returns:
        RecommendationBuilder instance
    """
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = RecommendationBuilder(logger)
    return _builder_instance
