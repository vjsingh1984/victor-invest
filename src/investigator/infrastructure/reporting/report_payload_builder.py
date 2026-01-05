"""
Report Payload Builder

Transforms raw synthesis agent output into normalized PDF report payloads.

This module solves the structural mismatch between synthesis agent outputs
(which wrap data in {'response': {...}}) and PDFReportGenerator expectations
(which expect flat recommendation dicts matching InvestmentRecommendation schema).

Key transformations:
- Unwrap LLM response wrappers
- Convert scores (0-100 → 0-10 scale)
- Normalize field names and structures
- Sanitize missing/invalid data
- Provide sensible defaults

Author: InvestiGator Team
Date: 2025-11-02
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReportDataContract:
    """
    Canonical data contract between synthesis agent and PDF generator.

    This ensures both sides understand the expected structure and prevents
    future regressions when either component changes.
    """

    # Core identification
    symbol: str
    timestamp: str

    # Recommendation
    recommendation: str  # 'strong buy', 'buy', 'hold', 'sell', 'strong sell'
    confidence: int  # 0-100

    # Scores (0-10 scale for PDF)
    composite_score: float
    fundamental_score: float
    technical_score: float
    value_score: float = 5.0
    growth_score: float = 5.0
    business_quality_score: float = 5.0

    # Financial metrics
    current_price: float = 0.0
    fair_value: float = 0.0
    price_target_12m: float = 0.0
    market_cap: float = 0.0

    # Investment thesis
    investment_thesis: str = ""
    key_insights: List[str] = field(default_factory=list)
    value_drivers: List[str] = field(default_factory=list)

    # Risk assessment
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    overall_risk: float = 50.0  # 0-100
    primary_risks: List[str] = field(default_factory=list)
    risk_tier: str = "MEDIUM"

    # Scenarios
    scenarios: Dict[str, Any] = field(default_factory=dict)
    bull_case: Optional[Dict] = None
    base_case: Optional[Dict] = None
    bear_case: Optional[Dict] = None

    # Action plan
    action_plan: Dict[str, Any] = field(default_factory=dict)
    specific_actions: List[str] = field(default_factory=list)

    # Trends and analysis
    multi_year_trends: Dict[str, Any] = field(default_factory=dict)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)

    # Conflicts and reconciliation
    conflicts: List[str] = field(default_factory=list)
    reconciliation: str = ""

    # Data quality
    data_quality_grade: str = "N/A"
    data_quality_score: float = 0.0

    # Charts
    chart_paths: List[str] = field(default_factory=list)


class ReportPayloadBuilder:
    """
    Builds normalized PDF report payloads from synthesis agent outputs.

    Handles all data transformations, validations, and fallbacks needed
    to convert raw synthesis data into PDF-ready recommendation dicts.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the payload builder."""
        self.logger = logger or logging.getLogger(__name__)

    def build(
        self,
        symbol: str,
        synthesis_report: Dict[str, Any],
        fundamental_data: Optional[Dict] = None,
        technical_data: Optional[Dict] = None,
        chart_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build normalized PDF report payload from synthesis output.

        Args:
            symbol: Stock symbol
            synthesis_report: Raw synthesis agent output
            fundamental_data: Optional fundamental analysis data for backfilling
            technical_data: Optional technical analysis data
            chart_paths: Optional list of chart file paths

        Returns:
            Normalized recommendation dict ready for PDFReportGenerator
        """
        self.logger.info(f"Building PDF payload for {symbol}")

        # Unwrap LLM response if needed
        unwrapped = self._unwrap_response(synthesis_report)

        # Extract core fields
        recommendation = self._extract_recommendation(unwrapped)
        scores = self._extract_scores(unwrapped)
        financials = self._extract_financials(unwrapped, fundamental_data, technical_data)
        thesis = self._extract_thesis(unwrapped)
        risks = self._extract_risks(unwrapped)
        scenarios = self._extract_scenarios(unwrapped)
        action_plan = self._extract_action_plan(unwrapped)

        # Backfill missing critical fields from fundamental and technical data
        if fundamental_data:
            financials = self._backfill_financials(financials, fundamental_data)
            scores = self._backfill_scores(scores, fundamental_data)

        # Backfill from technical data if still missing
        if technical_data:
            financials = self._backfill_from_technical(financials, technical_data)

        # Validate and sanitize
        self._validate_payload(symbol, financials, scores)

        # Build normalized payload
        payload = {
            "symbol": symbol,
            "timestamp": synthesis_report.get("timestamp", ""),
            # Recommendation
            "recommendation": recommendation.get("action", "hold"),
            "confidence": int(recommendation.get("confidence", 50)),
            # Scores (convert to 0-10 scale)
            "composite_score": self._scale_score(scores.get("composite", 50)),
            "fundamental_score": self._scale_score(scores.get("fundamental", 50)),
            "technical_score": self._scale_score(scores.get("technical", 50)),
            "value_score": self._scale_score(scores.get("value", 50)),
            "growth_score": self._scale_score(scores.get("growth", 50)),
            "business_quality_score": self._scale_score(scores.get("quality", 50)),
            # Financials
            "current_price": financials.get("current_price", 0),
            "fair_value": financials.get("fair_value", 0),
            "price_target_12m": financials.get("price_target_12m", 0),
            "market_cap": financials.get("market_cap", 0),
            # Investment thesis
            "investment_thesis": thesis.get("thesis", ""),
            "key_insights": thesis.get("insights", []),
            "value_drivers": thesis.get("drivers", []),
            # Risk assessment
            "risk_assessment": risks,
            "overall_risk": int(risks.get("overall_risk", 50)),
            "primary_risks": risks.get("primary_risks", []),
            "risk_tier": risks.get("tier", "MEDIUM"),
            # Scenarios
            "scenarios": scenarios,
            "bull_case": scenarios.get("bull_case"),
            "base_case": scenarios.get("base_case"),
            "bear_case": scenarios.get("bear_case"),
            # Action plan
            "action_plan": action_plan,
            "specific_actions": action_plan.get("actions", []),
            # Trends
            "multi_year_trends": unwrapped.get("multi_year_trends", {}),
            "trend_analysis": unwrapped.get("trend_analysis", {}),
            # Conflicts
            "conflicts": unwrapped.get("conflicts", []),
            "reconciliation": unwrapped.get("reconciliation", ""),
            # Charts
            "chart_paths": chart_paths or [],
        }

        self.logger.info(
            f"✅ Built payload for {symbol}: {recommendation.get('action', 'hold').upper()} "
            f"(composite: {payload['composite_score']:.1f}/10, "
            f"price: ${payload['current_price']:.2f})"
        )

        return payload

    def _unwrap_response(self, synthesis_report: Dict) -> Dict:
        """Unwrap LLM response wrappers to get actual data."""
        # Check for nested response wrapper
        if "response" in synthesis_report:
            data = synthesis_report["response"]
            if isinstance(data, dict):
                # Further unwrap if there's a 'report' key
                if "report" in data:
                    return data["report"]
                return data
            else:
                self.logger.warning(f"Response is {type(data)}, not dict - returning empty")
                return {}
        return synthesis_report

    def _extract_recommendation(self, data: Dict) -> Dict:
        """Extract recommendation and confidence."""
        rec = data.get("recommendation", {})
        if isinstance(rec, dict):
            return {
                "action": rec.get("action", rec.get("recommendation", "hold")),
                "confidence": rec.get("confidence", rec.get("confidence_level", 50)),
            }
        return {"action": "hold", "confidence": 50}

    def _extract_scores(self, data: Dict) -> Dict:
        """
        Extract all scores from synthesis response.

        Handles multiple field naming conventions:
        - composite_scores.overall_score → composite
        - composite_scores.fundamental_score → fundamental
        - Fallback to analysis_scores if composite_scores missing
        """
        # Try composite_scores first (current synthesis format)
        composite = data.get("composite_scores", {})
        analysis = data.get("analysis_scores", {})

        # Try to get assessment scores as fallback
        assessment = data.get("fundamental_assessment", {})

        return {
            "composite": (
                composite.get("overall_score") or composite.get("composite") or analysis.get("composite", 50)
            ),
            "fundamental": (
                composite.get("fundamental_score")
                or analysis.get("fundamental")
                or assessment.get("financial_health", {}).get("score", 50)
            ),
            "technical": (composite.get("technical_score") or analysis.get("technical", 50)),
            "value": (composite.get("value_score") or analysis.get("value", 50)),
            "growth": (composite.get("growth_score") or analysis.get("growth", 50)),
            "quality": (
                composite.get("business_quality_score") or composite.get("quality_score") or analysis.get("quality", 50)
            ),
        }

    def _extract_financials(self, data: Dict, fundamental_data: Optional[Dict], technical_data: Optional[Dict]) -> Dict:
        """
        Extract financial metrics with comprehensive fallback chain.

        Priority:
        1. Structured synthesis data (valuation dict)
        2. Narrative report appendix (key_metrics_summary)
        3. Fundamental agent data
        """
        valuation = data.get("valuation", {})

        # Try narrative report appendix (where synthesis actually puts the data)
        appendix = data.get("appendix", {})
        key_metrics = appendix.get("key_metrics_summary", {})

        # Helper to parse currency strings like "$270.37" -> 270.37
        def parse_currency(value):
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                # Remove $, commas, B/M/K suffixes
                cleaned = value.replace("$", "").replace(",", "").strip()
                # Handle B/M/K suffixes
                multiplier = 1
                if cleaned.endswith("B"):
                    multiplier = 1_000_000_000
                    cleaned = cleaned[:-1]
                elif cleaned.endswith("M"):
                    multiplier = 1_000_000
                    cleaned = cleaned[:-1]
                elif cleaned.endswith("K"):
                    multiplier = 1_000
                    cleaned = cleaned[:-1]
                try:
                    return float(cleaned) * multiplier
                except ValueError:
                    return 0
            return 0

        # Extract with fallback chain
        financials = {
            "current_price": (
                valuation.get("current_price", 0) or parse_currency(key_metrics.get("current_price", 0)) or 0
            ),
            "fair_value": (valuation.get("fair_value", 0) or parse_currency(key_metrics.get("fair_value", 0)) or 0),
            "price_target_12m": (
                valuation.get("price_target_12m", 0)
                or valuation.get("price_target", 0)
                or parse_currency(key_metrics.get("price_target", 0))
                or 0
            ),
            "market_cap": (data.get("market_cap", 0) or parse_currency(key_metrics.get("market_cap", 0)) or 0),
        }

        # Backfill from fundamental if still missing
        if fundamental_data:
            fund_val = fundamental_data.get("valuation", {})
            fund_analysis_response = fundamental_data.get("analysis", {}).get("response", {})
            fund_ratios = fund_analysis_response.get("ratios", {})
            fund_company_data = fundamental_data.get("analysis", {}).get("company_data", {})

            if financials["current_price"] == 0:
                financials["current_price"] = (
                    fund_val.get("current_price", 0)
                    or fund_ratios.get("current_price", 0)
                    or fund_company_data.get("current_price", 0)
                )
            if financials["fair_value"] == 0:
                financials["fair_value"] = fund_val.get("fair_value", 0)
            if financials["market_cap"] == 0:
                financials["market_cap"] = (
                    fundamental_data.get("market_cap", 0)
                    or fund_company_data.get("market_cap", 0)
                    or fund_ratios.get("market_cap", 0)
                )

        return financials

    def _extract_thesis(self, data: Dict) -> Dict:
        """
        Extract investment thesis and insights.

        Handles multiple response structures:
        - executive_summary.investment_thesis (current format)
        - investment_thesis (legacy format)
        - Nested dict with summary/thesis keys
        """
        # Try executive_summary first (current synthesis format)
        exec_summary = data.get("executive_summary", {})
        thesis_data = data.get("investment_thesis", {})

        # Handle both string and dict formats
        if exec_summary and isinstance(exec_summary, dict):
            # Current format: executive_summary contains thesis
            thesis = exec_summary.get("investment_thesis", "")
            insights = data.get("key_insights", [])
            drivers = []  # Not always present in executive_summary
        elif isinstance(thesis_data, str):
            # Legacy format: thesis is a direct string
            thesis = thesis_data
            insights = data.get("key_insights", [])
            drivers = []
        elif isinstance(thesis_data, dict):
            # Legacy format: thesis is nested dict
            thesis = thesis_data.get("summary", thesis_data.get("thesis", ""))
            insights = thesis_data.get("key_insights", thesis_data.get("insights", []))
            drivers = thesis_data.get("value_drivers", thesis_data.get("drivers", []))
        else:
            thesis = ""
            insights = []
            drivers = []

        # If still empty, try fundamental_assessment
        if not thesis:
            fund_assessment = data.get("fundamental_assessment", {})
            if isinstance(fund_assessment, dict):
                thesis = fund_assessment.get("investment_thesis", "")

        return {
            "thesis": thesis,
            "insights": insights if isinstance(insights, list) else [],
            "drivers": drivers if isinstance(drivers, list) else [],
        }

    def _extract_risks(self, data: Dict) -> Dict:
        """Extract risk assessment."""
        risk_data = data.get("risk_assessment", {})

        return {
            "overall_risk": risk_data.get("overall_risk", risk_data.get("risk_score", 50)),
            "primary_risks": risk_data.get("primary_risks", risk_data.get("risks", [])),
            "tier": risk_data.get("risk_tier", risk_data.get("tier", "MEDIUM")),
        }

    def _extract_scenarios(self, data: Dict) -> Dict:
        """Extract price scenarios."""
        scenarios = data.get("scenarios", {})

        return {
            "bull_case": scenarios.get("bull_case", scenarios.get("bull")),
            "base_case": scenarios.get("base_case", scenarios.get("base")),
            "bear_case": scenarios.get("bear_case", scenarios.get("bear")),
        }

    def _extract_action_plan(self, data: Dict) -> Dict:
        """Extract action plan."""
        action_plan = data.get("action_plan", {})

        actions = action_plan.get("specific_actions", action_plan.get("actions", []))

        return {
            "actions": actions if isinstance(actions, list) else [],
            "timeframe": action_plan.get("timeframe", ""),
            "monitoring": action_plan.get("monitoring", []),
        }

    def _backfill_financials(self, financials: Dict, fundamental_data: Dict) -> Dict:
        """Backfill missing financials from fundamental analysis."""
        if financials["current_price"] == 0:
            # Try ratios
            ratios = fundamental_data.get("analysis", {}).get("response", {}).get("ratios", {})
            financials["current_price"] = ratios.get("current_price", 0)

        if financials["market_cap"] == 0:
            # Try multiple locations including company_data
            ratios = fundamental_data.get("analysis", {}).get("response", {}).get("ratios", {})
            company_data = fundamental_data.get("analysis", {}).get("company_data", {})
            financials["market_cap"] = (
                fundamental_data.get("market_cap", 0)
                or company_data.get("market_cap", 0)
                or ratios.get("market_cap", 0)
                or fundamental_data.get("analysis", {}).get("response", {}).get("market_cap", 0)
            )
            if financials["market_cap"] > 0:
                self.logger.info(f"✅ Backfilled market_cap from fundamental agent: ${financials['market_cap']:,.0f}")

        return financials

    def _backfill_scores(self, scores: Dict, fundamental_data: Dict) -> Dict:
        """Backfill scores from fundamental data."""
        # If composite is still default, try to calculate from fundamentals
        if scores["composite"] == 50:
            quality_score = fundamental_data.get("data_quality", {}).get("data_quality_score", 0)
            if quality_score > 0:
                # Use quality score as a proxy for composite
                scores["composite"] = quality_score

        return scores

    def _backfill_from_technical(self, financials: Dict, technical_data: Dict) -> Dict:
        """
        Backfill missing financials from technical analysis.

        Technical agent has current_price in: technical['analysis']['response']['current_price']
        """
        # Extract from technical analysis response
        tech_analysis = technical_data.get("analysis", {}).get("response", {})

        if financials["current_price"] == 0:
            current_price = tech_analysis.get("current_price", 0)
            if current_price > 0:
                financials["current_price"] = current_price
                self.logger.info(f"✅ Backfilled current_price from technical agent: ${current_price:.2f}")

        # Technical analysis might also have market_cap in some cases
        if financials["market_cap"] == 0:
            market_cap = tech_analysis.get("market_cap", 0)
            if market_cap > 0:
                financials["market_cap"] = market_cap
                self.logger.info(f"✅ Backfilled market_cap from technical agent: ${market_cap:,.0f}")

        return financials

    def _scale_score(self, score: float) -> float:
        """
        Convert score to 0-10 scale.

        Handles both 0-100 and 0-10 scale inputs intelligently:
        - If score > 10, assume 0-100 scale and divide by 10
        - If score <= 10, assume already on 0-10 scale
        """
        if score > 10:
            # Score is on 0-100 scale, convert to 0-10
            return round(float(score) / 10, 1)
        else:
            # Score is already on 0-10 scale
            return round(float(score), 1)

    def _validate_payload(self, symbol: str, financials: Dict, scores: Dict):
        """Validate critical fields and log warnings."""
        issues = []

        if financials["current_price"] == 0:
            issues.append("current_price=0")

        if financials["market_cap"] == 0:
            issues.append("market_cap=0")

        if scores["composite"] == 50:
            issues.append("composite_score=default(50)")

        if issues:
            self.logger.warning(f"⚠️  Payload validation issues for {symbol}: {', '.join(issues)}")
