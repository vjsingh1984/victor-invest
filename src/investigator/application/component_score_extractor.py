"""
Component Score Extractor - Extracted from InvestmentSynthesizer for SRP.

This module handles extraction of individual component scores from LLM responses:
- Income statement score
- Cash flow score
- Balance sheet score
- Growth prospects score
- Value investment score
- Business quality score

Part of Phase 5 refactoring to break up monolithic synthesizer.py.

Author: InvestiGator Team
Date: 2025-01-05
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ComponentScoreExtractor:
    """
    Extracts individual component scores from LLM analysis responses.

    Extracted from InvestmentSynthesizer to follow Single Responsibility Principle.
    All component score extraction logic is centralized here.

    Component scores include:
    - Income: Profitability and margin analysis
    - Cashflow: Operating cash flow and liquidity
    - Balance: Asset/liability health and leverage
    - Growth: Revenue and expansion prospects
    - Value: Valuation relative to peers
    - Business Quality: Competitive moat and operational excellence
    """

    def __init__(
        self,
        fundamental_score_calculator: Optional[Callable[[Dict], float]] = None,
    ):
        """
        Initialize component score extractor.

        Args:
            fundamental_score_calculator: Optional callable to calculate fundamental score.
                                         If not provided, returns 0.0 for fallback cases.
        """
        self._calculate_fundamental_score = fundamental_score_calculator or (lambda _: 0.0)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def extract_income_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """
        Extract income statement score from responses.

        Extraction order:
        1. Direct score from AI recommendation
        2. Calculate from profitability margins in comprehensive analysis
        3. Fallback to fundamental score * 0.9

        Args:
            llm_responses: Dictionary of LLM analysis responses
            ai_recommendation: AI recommendation dictionary

        Returns:
            Income statement score (0.0-10.0)
        """
        # First check AI recommendation
        if "income_statement_score" in ai_recommendation:
            return float(ai_recommendation["income_statement_score"])

        # Check comprehensive analysis for income statement analysis
        comp_analysis = llm_responses.get("fundamental", {}).get("comprehensive", {})
        content = comp_analysis.get("content", comp_analysis) if isinstance(comp_analysis, dict) else {}

        if isinstance(content, dict):
            # Look for income statement analysis section
            income_analysis = content.get("income_statement_analysis", {})
            if income_analysis:
                # Try to extract a score from profitability metrics
                profitability = income_analysis.get("profitability_analysis", {})
                margins = [
                    profitability.get("gross_margin", 0),
                    profitability.get("operating_margin", 0),
                    profitability.get("net_margin", 0),
                ]
                # Convert margins to score (assuming good margins are >15%)
                avg_margin = (
                    sum(m for m in margins if m > 0) / len([m for m in margins if m > 0])
                    if any(m > 0 for m in margins)
                    else 0
                )
                if avg_margin > 0:
                    return min(10.0, max(1.0, avg_margin * 100 / 3))  # Scale to 1-10

        # Fallback to fundamental score with adjustment
        base_fundamental = self._calculate_fundamental_score(llm_responses)
        return base_fundamental * 0.9 if base_fundamental > 0 else 0.0

    def extract_cashflow_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """
        Extract cash flow score from responses.

        Analyzes keyword frequency for cash flow related terms and adjusts
        the fundamental score accordingly.

        Args:
            llm_responses: Dictionary of LLM analysis responses
            ai_recommendation: AI recommendation dictionary

        Returns:
            Cash flow score (0.0-10.0)
        """
        base_fundamental = self._calculate_fundamental_score(llm_responses)

        # Look for cash flow keywords
        cashflow_keywords = ["cash flow", "cash", "liquidity", "fcf", "working capital", "operating cash"]
        cashflow_score_adjustments = []

        for resp in llm_responses.get("fundamental", {}).values():
            content = resp.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            content = content.lower()
            cashflow_mentions = sum(1 for keyword in cashflow_keywords if keyword in content)
            if cashflow_mentions > 3:
                cashflow_score_adjustments.append(0.5)
            elif cashflow_mentions > 0:
                cashflow_score_adjustments.append(0.0)
            else:
                cashflow_score_adjustments.append(-0.5)

        adjustment = (
            sum(cashflow_score_adjustments) / len(cashflow_score_adjustments) if cashflow_score_adjustments else 0
        )
        return max(0.0, min(10.0, base_fundamental + adjustment)) if base_fundamental > 0 else 0.0

    def extract_balance_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """
        Extract balance sheet score from responses.

        Analyzes keyword frequency for balance sheet related terms and adjusts
        the fundamental score accordingly.

        Args:
            llm_responses: Dictionary of LLM analysis responses
            ai_recommendation: AI recommendation dictionary

        Returns:
            Balance sheet score (0.0-10.0)
        """
        base_fundamental = self._calculate_fundamental_score(llm_responses)

        # Look for balance sheet keywords
        balance_keywords = ["asset", "liability", "equity", "debt", "balance sheet", "leverage", "solvency"]
        balance_score_adjustments = []

        for resp in llm_responses.get("fundamental", {}).values():
            content = resp.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            content = content.lower()
            balance_mentions = sum(1 for keyword in balance_keywords if keyword in content)
            if balance_mentions > 3:
                balance_score_adjustments.append(0.5)
            elif balance_mentions > 0:
                balance_score_adjustments.append(0.0)
            else:
                balance_score_adjustments.append(-0.5)

        adjustment = sum(balance_score_adjustments) / len(balance_score_adjustments) if balance_score_adjustments else 0
        return max(0.0, min(10.0, base_fundamental + adjustment)) if base_fundamental > 0 else 0.0

    def extract_growth_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """
        Extract growth prospects score from responses.

        Extraction order:
        1. Direct score from comprehensive fundamental analysis
        2. Score from AI recommendation
        3. Keyword-based adjustment

        Args:
            llm_responses: Dictionary of LLM analysis responses
            ai_recommendation: AI recommendation dictionary

        Returns:
            Growth prospects score (0.0-10.0)
        """
        # First check if growth score is in the comprehensive fundamental analysis
        if "comprehensive" in llm_responses.get("fundamental", {}):
            comp_content = llm_responses["fundamental"]["comprehensive"].get("content", {})
            if isinstance(comp_content, dict) and "growth_prospects_score" in comp_content:
                return float(comp_content["growth_prospects_score"])

        # Check AI recommendation for growth assessment
        if "fundamental_assessment" in ai_recommendation:
            fund_assess = ai_recommendation["fundamental_assessment"]
            if "growth_prospects" in fund_assess:
                # Extract numeric score if available
                growth_data = fund_assess["growth_prospects"]
                if isinstance(growth_data, dict) and "score" in growth_data:
                    return float(growth_data["score"])

        # Fallback: analyze growth keywords
        base_fundamental = self._calculate_fundamental_score(llm_responses)
        growth_keywords = ["growth", "expansion", "increase", "momentum", "acceleration", "scaling"]
        growth_score_adjustments = []

        for resp in llm_responses.get("fundamental", {}).values():
            content = resp.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            content = content.lower()
            growth_mentions = sum(1 for keyword in growth_keywords if keyword in content)
            if growth_mentions > 5:
                growth_score_adjustments.append(1.0)
            elif growth_mentions > 2:
                growth_score_adjustments.append(0.5)
            else:
                growth_score_adjustments.append(0.0)

        adjustment = sum(growth_score_adjustments) / len(growth_score_adjustments) if growth_score_adjustments else 0
        return max(0.0, min(10.0, base_fundamental + adjustment)) if base_fundamental > 0 else 0.0

    def extract_value_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """
        Extract value investment score from responses.

        Analyzes valuation-related keywords (undervalued vs overvalued) and
        adjusts the fundamental score accordingly.

        Args:
            llm_responses: Dictionary of LLM analysis responses
            ai_recommendation: AI recommendation dictionary

        Returns:
            Value investment score (0.0-10.0)
        """
        # Check for valuation metrics in AI recommendation
        if "fundamental_assessment" in ai_recommendation:
            fund_assess = ai_recommendation["fundamental_assessment"]
            if "valuation" in fund_assess:
                val_data = fund_assess["valuation"]
                if isinstance(val_data, dict) and "score" in val_data:
                    return float(val_data["score"])

        # Look for value indicators
        base_fundamental = self._calculate_fundamental_score(llm_responses)
        value_keywords = ["undervalued", "discount", "cheap", "value", "pe ratio", "price to book", "dividend yield"]
        negative_value_keywords = ["overvalued", "expensive", "premium", "overpriced"]
        value_score_adjustments = []

        for resp in llm_responses.get("fundamental", {}).values():
            content = resp.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            content = content.lower()

            value_mentions = sum(1 for keyword in value_keywords if keyword in content)
            negative_mentions = sum(1 for keyword in negative_value_keywords if keyword in content)

            net_value_signal = value_mentions - negative_mentions
            if net_value_signal > 3:
                value_score_adjustments.append(1.0)
            elif net_value_signal > 0:
                value_score_adjustments.append(0.5)
            elif net_value_signal < -3:
                value_score_adjustments.append(-1.0)
            else:
                value_score_adjustments.append(0.0)

        adjustment = sum(value_score_adjustments) / len(value_score_adjustments) if value_score_adjustments else 0
        return max(0.0, min(10.0, base_fundamental + adjustment)) if base_fundamental > 0 else 0.0

    def extract_business_quality_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """
        Extract business quality score from SEC comprehensive analysis.

        This method works backwards from the comprehensive SEC analysis which aggregates
        quarterly data to assess business quality based on:
        - Core business concepts and tags identified across quarters
        - Revenue quality and consistency patterns
        - Operational efficiency metrics
        - Competitive positioning indicators
        - Management effectiveness signals

        Args:
            llm_responses: Dictionary of LLM analysis responses
            ai_recommendation: AI recommendation dictionary

        Returns:
            Business quality score (0.0-10.0), or 0.0 if not available
        """
        # First, try to get the business_quality_score directly from SEC comprehensive analysis
        comprehensive_analysis = llm_responses.get("fundamental", {}).get("comprehensive", {})
        if isinstance(comprehensive_analysis, dict):
            # Direct score from comprehensive analysis
            if "business_quality_score" in comprehensive_analysis:
                score_data = comprehensive_analysis["business_quality_score"]
                if isinstance(score_data, dict):
                    return float(score_data.get("score", 5.0))
                return float(score_data)

            # Extract from response content if it's nested
            content = comprehensive_analysis.get("content", {})
            if isinstance(content, dict) and "business_quality_score" in content:
                score_data = content["business_quality_score"]
                if isinstance(score_data, dict):
                    return float(score_data.get("score", 5.0))
                return float(score_data)

        # If comprehensive analysis is available as string/JSON, parse it
        if isinstance(comprehensive_analysis, str):
            try:
                parsed = json.loads(comprehensive_analysis)
                if "business_quality_score" in parsed:
                    return float(parsed["business_quality_score"])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: Calculate from quarterly analyses patterns
        quarterly_analyses = llm_responses.get("fundamental", {})
        quality_indicators = []

        for period_key, analysis in quarterly_analyses.items():
            if period_key == "comprehensive":  # Skip the comprehensive entry
                continue

            content = analysis.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)

            # Analyze quarterly data for business quality indicators
            quality_score = self.analyze_quarterly_business_quality(content, period_key)
            if quality_score > 0:
                quality_indicators.append(quality_score)

        # Calculate average business quality from quarterly analyses
        if quality_indicators:
            avg_quality = sum(quality_indicators) / len(quality_indicators)

            # Apply weighting based on data consistency and trends
            consistency_bonus = self.calculate_consistency_bonus(quality_indicators)
            final_score = min(10.0, max(1.0, avg_quality + consistency_bonus))

            return final_score

        # Ultimate fallback: Return 0 to indicate no business quality score available
        return 0.0

    def analyze_quarterly_business_quality(self, content: str, period: str) -> float:
        """
        Analyze individual quarterly content for business quality indicators.

        Evaluates content across four weighted categories:
        1. Revenue quality (1.5x weight)
        2. Operational excellence (1.2x weight)
        3. Innovation capacity (1.0x weight)
        4. Management effectiveness (0.8x weight)

        Args:
            content: Quarterly analysis content as string
            period: Period identifier (e.g., "Q1_2024")

        Returns:
            Business quality score (1.0-10.0)
        """
        content_lower = content.lower()
        quality_score = 5.0  # Base score

        # Revenue quality indicators
        revenue_quality_keywords = [
            "recurring revenue",
            "subscription",
            "diversified revenue",
            "stable revenue",
            "revenue growth",
            "market share",
            "competitive advantage",
            "moat",
        ]

        # Operational excellence indicators
        operational_keywords = [
            "margin expansion",
            "efficiency",
            "productivity",
            "automation",
            "cost control",
            "operating leverage",
            "scalability",
        ]

        # Innovation and competitive position
        innovation_keywords = [
            "innovation",
            "r&d",
            "research and development",
            "patent",
            "technology",
            "differentiation",
            "competitive position",
            "market leadership",
        ]

        # Management effectiveness
        management_keywords = [
            "capital allocation",
            "strategic initiative",
            "execution",
            "guidance",
            "shareholder value",
            "dividend",
            "buyback",
            "investment",
        ]

        # Calculate weighted scores for each category
        categories = [
            (revenue_quality_keywords, 1.5),  # Revenue quality most important
            (operational_keywords, 1.2),  # Operational efficiency
            (innovation_keywords, 1.0),  # Innovation capacity
            (management_keywords, 0.8),  # Management effectiveness
        ]

        total_weight = 0
        weighted_score = 0

        for keywords, weight in categories:
            category_score = 0
            for keyword in keywords:
                if keyword in content_lower:
                    category_score += 1

            # Normalize category score to 0-10 scale
            normalized_score = min(10.0, (category_score / len(keywords)) * 10)
            weighted_score += normalized_score * weight
            total_weight += weight

        # Calculate final weighted average
        if total_weight > 0:
            quality_score = weighted_score / total_weight

        return max(1.0, min(10.0, quality_score))

    def calculate_consistency_bonus(self, quality_indicators: List[float]) -> float:
        """
        Calculate bonus for consistent business quality across quarters.

        Lower standard deviation = more consistent = higher bonus.
        Maximum bonus is 1.0 point.

        Args:
            quality_indicators: List of quality scores across periods

        Returns:
            Consistency bonus (0.0-1.0)
        """
        if len(quality_indicators) < 2:
            return 0.0

        # Calculate standard deviation
        mean_quality = sum(quality_indicators) / len(quality_indicators)
        variance = sum((x - mean_quality) ** 2 for x in quality_indicators) / len(quality_indicators)
        std_dev = variance ** 0.5

        # Lower standard deviation = more consistent = higher bonus
        # Scale: 0-1 point bonus based on consistency
        max_bonus = 1.0
        consistency_bonus = max(0.0, max_bonus - (std_dev / 2.0))

        return consistency_bonus


# Singleton instance
_extractor_instance: Optional[ComponentScoreExtractor] = None


def get_component_score_extractor(
    fundamental_score_calculator: Optional[Callable[[Dict], float]] = None,
) -> ComponentScoreExtractor:
    """
    Get singleton ComponentScoreExtractor instance.

    Args:
        fundamental_score_calculator: Optional callable for fundamental score
                                     (only used on first call)

    Returns:
        ComponentScoreExtractor instance
    """
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ComponentScoreExtractor(fundamental_score_calculator)
    return _extractor_instance
