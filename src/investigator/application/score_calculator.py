"""
Score Calculator - Extracted from InvestmentSynthesizer for Single Responsibility.

This module handles all score calculation logic:
- Fundamental score extraction from LLM responses
- Technical score extraction from LLM responses
- Weighted overall score calculation
- Technical indicator extraction
- Momentum signal extraction
- Stop loss calculation

Part of Phase 5 refactoring to break up monolithic synthesizer.py.

Author: InvestiGator Team
Date: 2025-01-05
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScoreWeights:
    """Configuration for score weighting."""

    fundamental_weight: float = 0.6
    technical_weight: float = 0.4


class ScoreCalculator:
    """
    Calculates investment scores from LLM analysis responses.

    Extracted from InvestmentSynthesizer to follow Single Responsibility Principle.
    All score calculation logic is centralized here.
    """

    def __init__(self, weights: Optional[ScoreWeights] = None):
        """
        Initialize score calculator.

        Args:
            weights: Optional score weights configuration
        """
        self.weights = weights or ScoreWeights()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def calculate_fundamental_score(self, llm_responses: Dict) -> float:
        """
        Calculate fundamental score from LLM responses.

        Extraction logic:
        1. First try comprehensive analysis response
        2. Look for financial_health_score or overall_score
        3. Parse JSON string if needed
        4. Fallback to regex extraction
        5. Average quarterly scores if no comprehensive

        Args:
            llm_responses: Dictionary of LLM analysis responses

        Returns:
            Fundamental score (0.0-10.0), or 0.0 if not available
        """
        fundamental_responses = llm_responses.get("fundamental", {})
        if not fundamental_responses:
            return 0.0

        # First try to get from comprehensive analysis
        if "comprehensive" in fundamental_responses:
            comp_resp = fundamental_responses["comprehensive"]
            content = comp_resp.get("content", comp_resp)

            # Handle structured response
            if isinstance(content, dict):
                if "financial_health_score" in content:
                    return float(content["financial_health_score"])
                elif "overall_score" in content:
                    return float(content["overall_score"])

            # Handle string response
            elif isinstance(content, str):
                # Try to extract from JSON string
                try:
                    parsed = json.loads(content)
                    if "financial_health_score" in parsed:
                        return float(parsed["financial_health_score"])
                    elif "overall_score" in parsed:
                        return float(parsed["overall_score"])
                except (json.JSONDecodeError, ValueError):
                    # Fall back to regex
                    score_match = re.search(
                        r"(?:Financial Health|Overall|Score)[:\s]*(\d+(?:\.\d+)?)/10",
                        content,
                    )
                    if score_match:
                        return float(score_match.group(1))

        # If no comprehensive, try averaging quarterly scores
        scores = []
        for key, response in fundamental_responses.items():
            if key == "comprehensive":
                continue
            content = response.get("content", "")
            if isinstance(content, dict) and "financial_health_score" in content:
                scores.append(float(content["financial_health_score"]))
            elif isinstance(content, str):
                score_match = re.search(
                    r"(?:Financial Health|Overall|Score)[:\s]*(\d+(?:\.\d+)?)/10",
                    content,
                )
                if score_match:
                    scores.append(float(score_match.group(1)))

        return sum(scores) / len(scores) if scores else 0.0

    def calculate_technical_score(self, llm_responses: Dict) -> float:
        """
        Calculate technical score from structured JSON LLM response.

        Extraction logic:
        1. Look for technical_score in dict content
        2. Parse JSON string if needed
        3. Handle file format with === AI RESPONSE === header
        4. Fallback to regex for legacy format

        Args:
            llm_responses: Dictionary of LLM analysis responses

        Returns:
            Technical score (0.0-10.0), or 0.0 if not available
        """
        technical_response = llm_responses.get("technical")
        if not technical_response:
            return 0.0

        content = technical_response.get("content", "")

        # First try to parse as structured JSON (new format)
        if isinstance(content, dict):
            if "technical_score" in content:
                score_data = content["technical_score"]
                if isinstance(score_data, dict):
                    return float(score_data.get("score", 0.0))
                return float(score_data)

        elif isinstance(content, str):
            # Handle file format with headers
            json_content = content
            if "=== AI RESPONSE ===" in content:
                json_start = content.find("=== AI RESPONSE ===") + len("=== AI RESPONSE ===")
                json_content = content[json_start:].strip()

            try:
                parsed = json.loads(json_content)
                if "technical_score" in parsed:
                    score_data = parsed["technical_score"]
                    if isinstance(score_data, dict):
                        return float(score_data.get("score", 0.0))
                    return float(score_data)
            except json.JSONDecodeError:
                pass

            # Fall back to regex for legacy format
            score_match = re.search(
                r"(?:TECHNICAL[_\s]SCORE|technical_score)[:\s]*(\d+(?:\.\d+)?)",
                json_content,
                re.IGNORECASE,
            )
            if score_match:
                return float(score_match.group(1))

        return 0.0

    def calculate_weighted_score(
        self,
        fundamental_score: float,
        technical_score: float,
    ) -> float:
        """
        Calculate weighted overall score from fundamental and technical scores.

        Uses configurable weights with adjustments for extreme scores:
        - Extreme fundamental scores (>=8.5 or <=2.5) get 1.2x weight
        - Extreme technical scores (>=8.5 or <=2.5) get 1.1x weight

        Args:
            fundamental_score: Fundamental analysis score (0-10)
            technical_score: Technical analysis score (0-10)

        Returns:
            Weighted overall score (0-10)
        """
        if fundamental_score is None or technical_score is None:
            return 5.0

        fund_weight = self.weights.fundamental_weight
        tech_weight = self.weights.technical_weight

        # Adjust weights for extreme scores
        if fundamental_score >= 8.5 or fundamental_score <= 2.5:
            fund_weight *= 1.2

        if technical_score >= 8.5 or technical_score <= 2.5:
            tech_weight *= 1.1

        total_weight = fund_weight + tech_weight

        if total_weight == 0:
            return 0.0

        norm_fund_weight = fund_weight / total_weight
        norm_tech_weight = tech_weight / total_weight

        overall_score = fundamental_score * norm_fund_weight + technical_score * norm_tech_weight

        return round(overall_score, 1)

    def extract_technical_indicators(self, llm_responses: Dict) -> Dict:
        """
        Extract technical indicators from structured technical analysis JSON response.

        Args:
            llm_responses: Dictionary of LLM analysis responses

        Returns:
            Dictionary of extracted technical indicators
        """
        technical_response = llm_responses.get("technical")
        if not technical_response:
            return {}

        content = technical_response.get("content", "")
        indicators = {}

        # First try to parse as structured JSON
        if isinstance(content, dict):
            indicators = self._extract_indicators_from_dict(content)

        elif isinstance(content, str):
            indicators = self._extract_indicators_from_string(content)

        return indicators

    def _extract_indicators_from_dict(self, content: Dict) -> Dict:
        """Extract indicators from structured dict response."""
        return {
            "technical_score": content.get("technical_score", {}).get("score", 0.0),
            "trend_direction": content.get("trend_analysis", {}).get("primary_trend", "NEUTRAL"),
            "trend_strength": content.get("trend_analysis", {}).get("trend_strength", "WEAK"),
            "support_levels": [
                content.get("support_resistance", {}).get("immediate_support", 0.0),
                content.get("support_resistance", {}).get("major_support", 0.0),
            ],
            "resistance_levels": [
                content.get("support_resistance", {}).get("immediate_resistance", 0.0),
                content.get("support_resistance", {}).get("major_resistance", 0.0),
            ],
            "fibonacci_levels": content.get("support_resistance", {}).get("fibonacci_levels", {}),
            "momentum_signals": self.extract_momentum_signals(content),
            "risk_factors": content.get("risk_factors", []),
            "key_insights": content.get("key_insights", []),
            "catalysts": content.get("catalysts", []),
            "time_horizon": content.get("recommendation", {}).get("time_horizon", "MEDIUM"),
            "recommendation": content.get("recommendation", {}).get("technical_rating", "HOLD"),
            "confidence": content.get("recommendation", {}).get("confidence", "MEDIUM"),
            "position_sizing": content.get("recommendation", {}).get("position_sizing", "MODERATE"),
            "entry_strategy": content.get("entry_exit_strategy", {}),
            "volume_analysis": content.get("volume_analysis", {}),
            "volatility_analysis": content.get("volatility_analysis", {}),
            "sector_relative_strength": content.get("sector_relative_strength", {}),
        }

    def _extract_indicators_from_string(self, content: str) -> Dict:
        """Extract indicators from JSON string response."""
        try:
            # Handle file format with headers
            json_content = content
            if "=== AI RESPONSE ===" in content:
                json_start = content.find("=== AI RESPONSE ===") + len("=== AI RESPONSE ===")
                json_content = content[json_start:].strip()

            # Handle responses with <think> prefix
            json_start = json_content.find("{")
            if json_start >= 0:
                json_part = json_content[json_start:]
                # Find the end by counting braces
                brace_count = 0
                json_end = 0
                for i, char in enumerate(json_part):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

                if json_end > 0:
                    json_to_parse = json_part[:json_end]
                else:
                    json_to_parse = json_part

                parsed = json.loads(json_to_parse)
            else:
                parsed = json.loads(json_content)

            return {
                "technical_score": parsed.get("technical_score", 0.0),
                "trend_direction": parsed.get("trend_direction", "NEUTRAL"),
                "trend_strength": parsed.get("trend_strength", "WEAK"),
                "support_levels": parsed.get("support_levels", []),
                "resistance_levels": parsed.get("resistance_levels", []),
                "fibonacci_levels": parsed.get("support_resistance", {}).get("fibonacci_levels", {}),
                "momentum_signals": parsed.get("momentum_signals", []),
                "risk_factors": parsed.get("risk_factors", []),
                "key_insights": parsed.get("key_insights", []),
                "catalysts": parsed.get("catalysts", []),
                "time_horizon": parsed.get("time_horizon", "MEDIUM"),
                "recommendation": parsed.get("recommendation", "HOLD"),
                "confidence": parsed.get("confidence", "MEDIUM"),
                "position_sizing": "MODERATE",
                "entry_strategy": {},
                "volume_analysis": {},
                "volatility_analysis": {},
                "sector_relative_strength": {},
            }

        except json.JSONDecodeError:
            self.logger.debug("Failed to parse technical indicators from string")
            return {}

    def extract_momentum_signals(self, content: Dict) -> List[str]:
        """
        Extract momentum signals from technical analysis response.

        Args:
            content: Technical analysis content dictionary

        Returns:
            List of momentum signal descriptions
        """
        signals = []

        momentum = content.get("momentum_analysis", {})
        if momentum:
            # RSI signals
            rsi = momentum.get("rsi_14", 0)
            rsi_assessment = momentum.get("rsi_assessment", "")
            if rsi and rsi_assessment:
                signals.append(f"RSI ({rsi:.1f}) indicates {rsi_assessment.lower()} conditions")

            # MACD signals
            macd = momentum.get("macd", {})
            if macd.get("signal"):
                signals.append(f"MACD shows {macd['signal'].lower()} momentum")

            # Stochastic signals
            stoch = momentum.get("stochastic", {})
            if stoch.get("signal"):
                signals.append(f"Stochastic indicates {stoch['signal'].lower()} conditions")

        # Volume signals
        volume = content.get("volume_analysis", {})
        if volume.get("volume_trend"):
            signals.append(f"Volume trend is {volume['volume_trend'].lower()}")

        return signals

    def calculate_stop_loss(
        self,
        current_price: float,
        recommendation: Dict,
        overall_score: float,
    ) -> float:
        """
        Calculate stop loss based on risk management.

        Stop loss percentages by recommendation:
        - STRONG BUY: 12%
        - BUY: 10%
        - HOLD: 8%
        - SELL: 5%

        Adjusted for low conviction (score < 4.0): 0.5x multiplier

        Args:
            current_price: Current stock price
            recommendation: Recommendation dictionary with 'recommendation' key
            overall_score: Overall conviction score

        Returns:
            Stop loss price
        """
        if not current_price or current_price <= 0:
            return 0.0

        rec_type = recommendation.get("recommendation", "HOLD")

        if "STRONG BUY" in rec_type:
            stop_loss_pct = 0.12
        elif "BUY" in rec_type:
            stop_loss_pct = 0.10
        elif "HOLD" in rec_type:
            stop_loss_pct = 0.08
        else:  # SELL
            stop_loss_pct = 0.05

        # Adjust for overall score
        if overall_score < 4.0:
            stop_loss_pct *= 0.5

        return round(current_price * (1 - stop_loss_pct), 2)


# Singleton instance
_calculator_instance: Optional[ScoreCalculator] = None


def get_score_calculator(weights: Optional[ScoreWeights] = None) -> ScoreCalculator:
    """
    Get singleton ScoreCalculator instance.

    Args:
        weights: Optional score weights (only used on first call)

    Returns:
        ScoreCalculator instance
    """
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = ScoreCalculator(weights)
    return _calculator_instance
