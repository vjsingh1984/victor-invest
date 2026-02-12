"""Text-based insight and risk extraction helpers for synthesis outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


def extract_insights_from_text(text_details: str) -> Tuple[List[str], List[str]]:
    """Extract insight/risk bullets from freeform synthesis text."""
    insights: List[str] = []
    risks: List[str] = []

    if not text_details:
        return insights, risks

    text = text_details.strip()

    insight_patterns = [
        r"(?:key\s+)?insights?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)",
        r"(?:important\s+)?findings?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)",
        r"(?:notable\s+)?observations?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)",
        r"(?:investment\s+)?highlights?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)",
    ]
    for pattern in insight_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            bullet_items = re.findall(r"[•\-\*]\s*(.+)", match)
            numbered_items = re.findall(r"\d+\.\s*(.+)", match)
            for item in bullet_items + numbered_items:
                clean_item = item.strip()
                if len(clean_item) > 10 and clean_item not in insights:
                    insights.append(clean_item[:200])

    risk_patterns = [
        r"(?:key\s+)?risks?[:\s]+(.*?)(?=\n\n|\nkey\s+insights?|\n[A-Z]|$)",
        r"(?:risk\s+)?factors?[:\s]+(.*?)(?=\n\n|\nkey\s+insights?|\n[A-Z]|$)",
        r"(?:potential\s+)?concerns?[:\s]+(.*?)(?=\n\n|\nkey\s+insights?|\n[A-Z]|$)",
        r"(?:investment\s+)?risks?[:\s]+(.*?)(?=\n\n|\nkey\s+insights?|\n[A-Z]|$)",
        r"downside[:\s]+(.*?)(?=\n\n|\nkey\s+insights?|\n[A-Z]|$)",
    ]
    for pattern in risk_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            bullet_items = re.findall(r"[•\-\*]\s*(.+)", match)
            numbered_items = re.findall(r"\d+\.\s*(.+)", match)
            for item in bullet_items + numbered_items:
                clean_item = item.strip()
                if len(clean_item) > 10 and clean_item not in risks:
                    risks.append(clean_item[:200])

    if not insights and not risks:
        sentences = re.split(r"[.!?]+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            insight_indicators = ["strength", "opportunity", "advantage", "positive", "growth", "improve"]
            if any(indicator in sentence.lower() for indicator in insight_indicators) and len(insights) < 3:
                insights.append(sentence[:200])
            risk_indicators = ["risk", "concern", "challenge", "threat", "weakness", "decline", "pressure"]
            if any(indicator in sentence.lower() for indicator in risk_indicators) and len(risks) < 3:
                risks.append(sentence[:200])

    return insights[:5], risks[:5]


def _to_text(content: Any) -> str:
    if isinstance(content, dict):
        return json.dumps(content)
    if isinstance(content, str):
        return content
    return str(content)


def extract_comprehensive_risks(
    llm_responses: Dict[str, Any], ai_recommendation: Dict[str, Any], additional_risks: List[str] | None = None
) -> List[str]:
    """Extract and prioritize risk factors from synthesis + source responses."""
    risks: List[str] = []

    if isinstance(ai_recommendation, dict):
        ai_risks = ai_recommendation.get("key_risks", [])
        if isinstance(ai_risks, list):
            risks.extend(ai_risks[:3])

    if additional_risks:
        risks.extend(additional_risks)

    for resp in llm_responses.get("fundamental", {}).values():
        content = _to_text(resp.get("content", ""))
        risk_section = re.search(r"risk[s]?[:\s]*(.*?)(?=\n\n|\d+\.)", content, re.IGNORECASE | re.DOTALL)
        if risk_section:
            risk_items = re.findall(r"[•\-]\s*(.+)", risk_section.group(1))
            risks.extend(risk_items[:2])

    unique_risks: List[str] = []
    seen = set()
    for risk in risks:
        risk_lower = risk.lower().strip()
        if risk_lower not in seen and len(risk_lower) > 10:
            seen.add(risk_lower)
            unique_risks.append(risk)

    return unique_risks[:8] if unique_risks else ["Limited risk data available"]


def extract_comprehensive_insights(
    llm_responses: Dict[str, Any], ai_recommendation: Dict[str, Any], additional_insights: List[str] | None = None
) -> List[str]:
    """Extract and prioritize insights from synthesis + source responses."""
    insights: List[str] = []

    if additional_insights:
        insights.extend(additional_insights)

    if isinstance(ai_recommendation, dict):
        catalysts = ai_recommendation.get("key_catalysts", [])
        if isinstance(catalysts, list):
            insights.extend([f"Catalyst: {cat}" for cat in catalysts[:2]])

    for resp in llm_responses.get("fundamental", {}).values():
        content = _to_text(resp.get("content", ""))
        insights_section = re.search(
            r"key\s+(?:insight|finding)[s]?[:\s]*(.*?)(?=\n\n|\d+\.)", content, re.IGNORECASE | re.DOTALL
        )
        if insights_section:
            insight_items = re.findall(r"[•\-]\s*(.+)", insights_section.group(1))
            insights.extend(insight_items[:2])

    tech_resp = llm_responses.get("technical")
    if tech_resp:
        content = _to_text(tech_resp.get("content", ""))
        tech_insights = re.findall(
            r"KEY INSIGHTS[:\s]*\*?\*?(.*?)(?=\*\*[A-Z]|\n\n)", content, re.IGNORECASE | re.DOTALL
        )
        if tech_insights:
            tech_items = re.findall(r"[•\-]\s*(.+)", tech_insights[0])
            insights.extend([f"Technical: {item}" for item in tech_items[:2]])

    unique_insights: List[str] = []
    seen = set()
    for insight in insights:
        insight_lower = insight.lower().strip()
        if insight_lower not in seen and len(insight_lower) > 10:
            seen.add(insight_lower)
            unique_insights.append(insight)

    return unique_insights[:8] if unique_insights else ["Analysis insights pending"]
