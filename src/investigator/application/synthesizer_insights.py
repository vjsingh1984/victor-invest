"""
Helpers for synthesizer extensible-insights parsing and report shaping.

Extracted from ``InvestmentSynthesizer`` to keep orchestration logic separate
from text-inspection utilities.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

STANDARD_RECOMMENDATION_FIELDS = {
    "overall_score",
    "investment_thesis",
    "recommendation",
    "confidence_level",
    "position_size",
    "time_horizon",
    "risk_reward_ratio",
    "key_catalysts",
    "downside_risks",
    "thinking",
    "details",
    "processing_metadata",
    "_fallback_created",
    "_parsing_error",
}


def extract_reasoning_themes(thinking_content: str) -> List[str]:
    """Extract key reasoning themes from thinking content."""
    if not thinking_content:
        return []

    themes = []
    patterns = [
        r"fundamental[s]?\s+(?:analysis|factors|strengths)",
        r"technical\s+(?:analysis|indicators|patterns)",
        r"risk[s]?\s+(?:assessment|factors|considerations)",
        r"market\s+(?:position|environment|conditions)",
        r"valuation\s+(?:methods|approaches|metrics)",
        r"growth\s+(?:prospects|potential|drivers)",
        r"competitive\s+(?:advantage|position|landscape)",
    ]

    for pattern in patterns:
        if re.search(pattern, thinking_content, re.IGNORECASE):
            match = re.search(pattern, thinking_content, re.IGNORECASE)
            if match:
                themes.append(match.group(0).lower())

    return list(set(themes))


def extract_decision_process(thinking_content: str) -> Dict[str, Any]:
    """Extract decision-making process indicators from thinking content."""
    if not thinking_content:
        return {}

    return {
        "has_structured_approach": bool(re.search(r"first|then|next|finally", thinking_content, re.IGNORECASE)),
        "considers_alternatives": bool(
            re.search(r"but|however|alternatively|on the other hand", thinking_content, re.IGNORECASE)
        ),
        "weighs_factors": bool(re.search(r"weight|balance|consider|factor", thinking_content, re.IGNORECASE)),
        "mentions_uncertainty": bool(re.search(r"uncertain|unclear|maybe|might|could", thinking_content, re.IGNORECASE)),
        "shows_confidence": bool(re.search(r"confident|certain|sure|clear", thinking_content, re.IGNORECASE)),
    }


def detect_markdown_content(content: str) -> bool:
    """Detect whether content contains markdown formatting."""
    if not content:
        return False

    markdown_patterns = [r"\*\*.*?\*\*", r"\*.*?\*", r"#+ ", r"- ", r"\d+\. ", r"```"]
    return any(re.search(pattern, content) for pattern in markdown_patterns)


def extract_bullet_points(content: str) -> List[str]:
    """Extract bullet points from content."""
    if not content:
        return []

    patterns = [r"- (.+)", r"• (.+)", r"\* (.+)", r"\d+\. (.+)"]
    bullets = []

    for pattern in patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        bullets.extend(matches)

    return [bullet.strip() for bullet in bullets if len(bullet.strip()) > 5]


def extract_numerical_insights(content: str) -> List[Dict[str, Any]]:
    """Extract percentages, amounts, ratios, and multiples from content."""
    if not content:
        return []

    numbers = []
    patterns = [
        (r"(\d+(?:\.\d+)?%)", "percentage"),
        (r"\$(\d+(?:,\d{3})*(?:\.\d{2})?)", "monetary"),
        (r"(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", "ratio"),
        (r"(\d+(?:\.\d+)?x)", "multiple"),
    ]

    for pattern, num_type in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            numbers.append({"type": num_type, "value": match, "context": "content_extraction"})

    return numbers[:10]


def analyze_field_completeness(ai_recommendation: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze completeness of standard recommendation fields."""
    standard_fields = [
        "overall_score",
        "investment_thesis",
        "recommendation",
        "confidence_level",
        "position_size",
        "time_horizon",
        "risk_reward_ratio",
        "key_catalysts",
        "downside_risks",
    ]

    completeness = {
        "total_standard_fields": len(standard_fields),
        "present_fields": [],
        "missing_fields": [],
        "completeness_ratio": 0.0,
    }

    for field in standard_fields:
        if field in ai_recommendation and ai_recommendation[field]:
            completeness["present_fields"].append(field)
        else:
            completeness["missing_fields"].append(field)

    completeness["completeness_ratio"] = len(completeness["present_fields"]) / len(standard_fields)
    return completeness


def identify_custom_fields(ai_recommendation: Dict[str, Any]) -> List[str]:
    """Identify custom/non-standard fields in the recommendation."""
    custom_fields = []
    for key in ai_recommendation.keys():
        if key not in STANDARD_RECOMMENDATION_FIELDS:
            custom_fields.append(key)
    return custom_fields


def check_fallback_flags(ai_recommendation: Dict[str, Any]) -> Dict[str, bool]:
    """Check fallback and parsing flags."""
    return {
        "is_fallback": ai_recommendation.get("_fallback_created", False),
        "has_parsing_error": ai_recommendation.get("_parsing_error", False),
        "is_emergency_fallback": ai_recommendation.get("_emergency_fallback", False),
        "has_processing_metadata": bool(ai_recommendation.get("processing_metadata")),
    }


def recommend_report_sections(
    ai_recommendation: Dict[str, Any], thinking_content: str, additional_details: str
) -> List[str]:
    """Recommend report sections based on available content."""
    sections = ["executive_summary", "recommendation"]

    if thinking_content and len(thinking_content) > 200:
        sections.append("reasoning_analysis")

    if additional_details and len(additional_details) > 100:
        sections.append("additional_insights")

    if ai_recommendation.get("key_catalysts") and len(ai_recommendation["key_catalysts"]) > 2:
        sections.append("catalyst_analysis")

    if ai_recommendation.get("downside_risks") and len(ai_recommendation["downside_risks"]) > 2:
        sections.append("risk_assessment")

    if ai_recommendation.get("processing_metadata"):
        sections.append("methodology_notes")

    return sections


def extract_priority_insights(thinking_content: str, additional_details: str) -> List[str]:
    """Extract high-priority insights for report highlighting."""
    insights = []

    if thinking_content:
        key_phrases = re.findall(
            r"(?:key|important|crucial|critical|significant).{1,100}", thinking_content, re.IGNORECASE
        )
        insights.extend([phrase.strip() for phrase in key_phrases[:3]])

    if additional_details:
        highlights = re.findall(r"(?:\*\*|##).{1,100}", additional_details)
        insights.extend([highlight.strip() for highlight in highlights[:2]])

    return insights


def suggest_visualizations(ai_recommendation: Dict[str, Any]) -> List[str]:
    """Suggest visualizations based on recommendation content."""
    suggestions = []

    if ai_recommendation.get("overall_score"):
        suggestions.append("score_gauge_chart")

    if ai_recommendation.get("key_catalysts") and ai_recommendation.get("downside_risks"):
        suggestions.append("risk_catalyst_matrix")

    if ai_recommendation.get("time_horizon"):
        suggestions.append("timeline_chart")

    if ai_recommendation.get("processing_metadata", {}).get("tokens"):
        suggestions.append("processing_metrics_chart")

    return suggestions
