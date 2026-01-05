"""
Deterministic Analyzer - Extracted from FundamentalAnalysisAgent for SRP.

This module handles rule-based financial analysis without LLM calls:
- Financial health analysis (liquidity, solvency, capital structure)
- Growth analysis (revenue trends, earnings quality)
- Profitability analysis (margins, returns on capital)

All methods are deterministic - same inputs always produce same outputs.

Part of Phase 5 refactoring to break up monolithic agent.py.

Author: InvestiGator Team
Date: 2025-01-05
"""

import logging
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeterministicAnalyzer:
    """
    Performs rule-based financial analysis.

    Extracted from FundamentalAnalysisAgent to follow Single Responsibility Principle.
    All deterministic analysis logic is centralized here.

    Analysis types:
    - Financial Health: Liquidity, solvency, capital structure, working capital
    - Growth: Revenue sustainability, earnings quality, market share trends
    - Profitability: Margin trends, returns on capital, cost structure
    """

    def __init__(self, agent_id: str = "deterministic", logger: Optional[logging.Logger] = None):
        """
        Initialize deterministic analyzer.

        Args:
            agent_id: Identifier for this analyzer instance
            logger: Optional logger instance
        """
        self.agent_id = agent_id
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _build_deterministic_response(self, label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a structure consistent with LLM response format for rule-based analyses.

        Args:
            label: Analysis type label (e.g., "financial_health", "growth_analysis")
            payload: The analysis result payload

        Returns:
            Dictionary with response, model_info, and metadata
        """
        return {
            "response": payload,
            "prompt": "",
            "model_info": {
                "model": f"deterministic-{label}",
                "temperature": 0.0,
                "top_p": 0.0,
                "format": "json",
            },
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "analysis_type": label,
                "cache_type": "deterministic_analysis",
            },
        }

    def _require_financials(self, company_data: Dict) -> Dict:
        """
        Extract financials from company_data, returning empty dict if not present.

        Args:
            company_data: Company data dictionary

        Returns:
            Financials dictionary (may be empty)
        """
        return company_data.get("financials") or {}

    async def analyze_financial_health(self, company_data: Dict, ratios: Dict, symbol: str) -> Dict:
        """
        Evaluate liquidity, solvency, and working-capital resilience without LLM calls.

        Args:
            company_data: Company data with financials
            ratios: Calculated financial ratios
            symbol: Stock symbol

        Returns:
            Deterministic response with financial health assessment
        """
        financials = self._require_financials(company_data)

        def assess_liquidity() -> tuple:
            current_ratio = ratios.get("current_ratio")
            quick_ratio = ratios.get("quick_ratio")
            if current_ratio is None:
                return (
                    "Unknown",
                    "Current ratio unavailable; liquidity requires manual review.",
                    55.0,
                )
            if current_ratio >= 2.0:
                label, score = "Strong", 95.0
            elif current_ratio >= 1.2:
                label, score = "Adequate", 75.0
            else:
                label, score = "Weak", 45.0
            if quick_ratio:
                commentary = f"Current ratio {current_ratio:.2f}; quick ratio {quick_ratio:.2f}."
            else:
                commentary = f"Current ratio {current_ratio:.2f}."
            return label, commentary, score

        def assess_solvency() -> tuple:
            debt_to_equity = ratios.get("debt_to_equity")
            debt_to_assets = ratios.get("debt_to_assets")
            total_debt = financials.get("total_debt") or 0
            if total_debt == 0:
                return "Debt Free", "Company has no financial leverage.", 95.0
            if debt_to_equity is None:
                return "Unknown", "Debt-to-equity unavailable; solvency indeterminate.", 55.0
            if debt_to_equity <= 1.0:
                label, score = "Comfortable", 80.0
            elif debt_to_equity <= 2.0:
                label, score = "Managed", 65.0
            else:
                label, score = "Leveraged", 45.0

            def _fmt_ratio(value: Optional[float]) -> str:
                if value is None:
                    return "n/a"
                try:
                    return f"{value:.2f}"
                except (TypeError, ValueError):
                    return "n/a"

            commentary = f"Debt/Equity {debt_to_equity:.2f}; Debt/Assets {_fmt_ratio(debt_to_assets)}."
            return label, commentary, score

        def assess_capital_structure(sol_label: str, sol_score: float) -> tuple:
            net_debt = (financials.get("total_debt") or 0) - (financials.get("cash") or 0)
            if net_debt <= 0:
                return "Net Cash", "Cash reserves exceed total debt.", max(sol_score, 85.0)
            return sol_label, f"Net debt approximately ${net_debt:,.0f}.", sol_score

        def assess_working_capital() -> tuple:
            ocf = financials.get("operating_cash_flow")
            revenue = financials.get("revenues")
            if ocf is None or revenue in (None, 0):
                return "Mixed", "Operating cash-flow data unavailable.", 60.0
            ocf_margin = ocf / revenue if revenue else 0
            if ocf_margin >= 0.2:
                label, score = "Efficient", 85.0
            elif ocf_margin >= 0.1:
                label, score = "Stable", 70.0
            else:
                label, score = "Tight", 55.0
            commentary = f"OCF margin {ocf_margin:.1%}."
            return label, commentary, score

        def assess_debt_serviceability() -> tuple:
            interest_coverage = ratios.get("interest_coverage") or company_data.get("interest_coverage")
            total_debt = financials.get("total_debt") or 0
            if total_debt == 0:
                return "Not Applicable", "No debt outstanding.", 90.0
            if not interest_coverage:
                return "Unknown", "Interest coverage unavailable.", 55.0
            if interest_coverage >= 6:
                return "Comfortable", f"Coverage {interest_coverage:.1f}x.", 85.0
            if interest_coverage >= 2:
                return "Adequate", f"Coverage {interest_coverage:.1f}x.", 70.0
            return "Stressed", f"Coverage {interest_coverage:.1f}x.", 45.0

        def assess_flexibility() -> tuple:
            cash = financials.get("cash") or 0
            total_debt = financials.get("total_debt") or 0
            if total_debt == 0:
                return "High", "Balance sheet unlevered with cash cushion.", 90.0
            liquidity_ratio = (cash + (financials.get("short_term_investments") or 0)) / total_debt
            if liquidity_ratio >= 0.75:
                return "High", f"Liquid assets cover {liquidity_ratio:.0%} of debt.", 80.0
            if liquidity_ratio >= 0.4:
                return "Moderate", f"Liquid assets cover {liquidity_ratio:.0%} of debt.", 65.0
            return "Limited", f"Liquid assets cover {liquidity_ratio:.0%} of debt.", 50.0

        liquidity = assess_liquidity()
        solvency = assess_solvency()
        capital_structure = assess_capital_structure(solvency[0], solvency[2])
        working_capital = assess_working_capital()
        debt_serviceability = assess_debt_serviceability()
        flexibility = assess_flexibility()

        score_components = [
            liquidity[2],
            solvency[2],
            capital_structure[2],
            working_capital[2],
            debt_serviceability[2],
            flexibility[2],
        ]
        overall_health_score = round(sum(score_components) / len(score_components), 1)

        risk_factors: List[Dict[str, str]] = []
        if liquidity[0] == "Weak":
            risk_factors.append({"risk": "Tight liquidity", "commentary": liquidity[1]})
        if solvency[0] == "Leveraged":
            risk_factors.append({"risk": "High leverage", "commentary": solvency[1]})
        if debt_serviceability[0] == "Stressed":
            risk_factors.append({"risk": "Debt service pressure", "commentary": debt_serviceability[1]})

        payload = {
            "liquidity_position": {"assessment": liquidity[0], "commentary": liquidity[1]},
            "solvency": {"assessment": solvency[0], "commentary": solvency[1]},
            "capital_structure_quality": {"assessment": capital_structure[0], "commentary": capital_structure[1]},
            "working_capital_management": {"assessment": working_capital[0], "commentary": working_capital[1]},
            "debt_serviceability": {"assessment": debt_serviceability[0], "commentary": debt_serviceability[1]},
            "financial_flexibility": {"assessment": flexibility[0], "commentary": flexibility[1]},
            "risk_factors": risk_factors,
            "overall_health_score": overall_health_score,
        }

        return self._build_deterministic_response("financial_health", payload)

    async def analyze_growth(self, company_data: Dict, symbol: str) -> Dict:
        """
        Deterministic growth analysis leveraging computed trend data.

        Args:
            company_data: Company data with trend_analysis
            symbol: Stock symbol

        Returns:
            Deterministic response with growth assessment
        """
        trend = company_data.get("trend_analysis") or {}
        revenue_trend = trend.get("revenue") or {}
        margin_trend = trend.get("margins") or {}

        def _summarize_series(series: List[float]) -> Dict[str, Optional[float]]:
            if not series:
                return {"avg": None, "latest": None, "quantiles": {}}
            window = series[-min(len(series), 6) :]
            summary = {"avg": statistics.mean(window) if window else None, "latest": window[-1] if window else None}
            quantiles = {}
            if len(window) >= 4:
                try:
                    q1, q2, q3 = statistics.quantiles(window, n=4, method="inclusive")
                    quantiles = {"p25": q1, "p50": q2, "p75": q3}
                except statistics.StatisticsError:
                    quantiles = {}
            summary["quantiles"] = quantiles
            return summary

        yoy_summary = _summarize_series(revenue_trend.get("y_over_y_growth") or [])
        qoq_summary = _summarize_series(revenue_trend.get("q_over_q_growth") or [])
        comparisons = {
            "avg_yoy_growth": yoy_summary["avg"],
            "latest_qoq_growth": qoq_summary["latest"],
            "yoy_quantiles": yoy_summary["quantiles"],
            "qoq_quantiles": qoq_summary["quantiles"],
        }

        def classify_growth(value: Optional[float]) -> tuple:
            if value is None:
                return "Unknown", 60.0
            if value >= 8.0:
                return "High", 90.0
            if value >= 3.0:
                return "Moderate", 75.0
            if value >= 0.0:
                return "Stable", 65.0
            return "Contracting", 45.0

        yoy_growth = comparisons.get("avg_yoy_growth")
        qoq_growth = comparisons.get("latest_qoq_growth")
        yoy_label, yoy_score = classify_growth(yoy_growth)
        qoq_label, qoq_score = classify_growth(qoq_growth)

        consistency_score = revenue_trend.get("consistency_score")
        if consistency_score is None:
            consistency_label, consistency_pts = "Unknown", 60.0
        elif consistency_score >= 80:
            consistency_label, consistency_pts = "Low Volatility", 85.0
        elif consistency_score >= 60:
            consistency_label, consistency_pts = "Manageable", 70.0
        else:
            consistency_label, consistency_pts = "Choppy", 55.0

        margin_direction = margin_trend.get("net_margin_trend") or "stable"
        margin_map = {
            "expanding": ("Improving", 85.0),
            "stable": ("Stable", 70.0),
            "contracting": ("Weak", 55.0),
        }
        earnings_label, earnings_score = margin_map.get(margin_direction, ("Stable", 70.0))

        market_share_trend = revenue_trend.get("trend", "stable")
        market_map = {
            "accelerating": ("Gaining", 85.0),
            "stable": ("Holding", 70.0),
            "decelerating": ("Losing", 55.0),
        }
        market_label, market_score = market_map.get(market_share_trend, ("Holding", 70.0))

        growth_drivers: List[str] = []
        growth_risks: List[str] = []
        if yoy_growth and yoy_growth >= 5:
            growth_drivers.append("Product demand momentum")
        if margin_direction == "expanding":
            growth_drivers.append("Operational leverage")
        if trend.get("cyclical", {}).get("is_cyclical"):
            growth_risks.append("Seasonality swings")
        if yoy_growth is not None and yoy_growth < 0:
            growth_risks.append("Negative revenue comp")
        if not growth_drivers:
            growth_drivers.append("Core franchise stability")
        if not growth_risks:
            growth_risks.append("Execution risk")

        score_components = [yoy_score, qoq_score, consistency_pts, earnings_score, market_score]
        growth_score = round(sum(score_components) / len(score_components), 1)

        payload = {
            "revenue_growth_sustainability": {
                "assessment": yoy_label,
                "commentary": (
                    f"Average Y/Y growth {yoy_growth:.1f}%" if yoy_growth is not None else "Insufficient data"
                ),
            },
            "earnings_growth_quality": {
                "assessment": earnings_label,
                "commentary": f"Margin trend {margin_direction.upper()}.",
            },
            "growth_consistency_and_volatility": {
                "assessment": consistency_label,
                "commentary": (
                    f"Consistency score {consistency_score:.0f}/100"
                    if consistency_score is not None
                    else "Consistency data unavailable."
                ),
            },
            "market_share_trends": {
                "assessment": market_label,
                "commentary": f"Revenue trend classified as {market_share_trend}.",
            },
            "growth_drivers_and_catalysts": growth_drivers,
            "future_growth_potential": {
                "assessment": "High" if yoy_label in {"High", "Moderate"} else "Balanced",
                "commentary": (
                    "Pipeline supported by demand indicators."
                    if yoy_label in {"High", "Moderate"}
                    else "Requires catalysts to re-accelerate."
                ),
            },
            "growth_risks_and_headwinds": growth_risks,
            "distribution_snapshot": {
                "yoy_percentiles": comparisons.get("yoy_quantiles"),
                "qoq_percentiles": comparisons.get("qoq_quantiles"),
            },
            "growth_score": growth_score,
        }

        return self._build_deterministic_response("growth_analysis", payload)

    async def analyze_profitability(self, company_data: Dict, ratios: Dict, symbol: str) -> Dict:
        """
        Deterministic profitability assessment using core ratios.

        Args:
            company_data: Company data with trend_analysis
            ratios: Calculated financial ratios
            symbol: Stock symbol

        Returns:
            Deterministic response with profitability assessment
        """
        trend = company_data.get("trend_analysis") or {}
        margin_trend = trend.get("margins", {})

        def pct(value: Optional[float]) -> str:
            if value is None:
                return "n/a"
            return f"{value*100:.1f}%" if value <= 1 else f"{value:.1f}%"

        gross_margin = ratios.get("gross_margin")
        operating_margin = ratios.get("operating_margin")
        net_margin = ratios.get("net_margin")
        roe = ratios.get("roe")
        roa = ratios.get("roa")
        asset_turnover = ratios.get("asset_turnover")

        def classify_margin(value: Optional[float]) -> tuple:
            if value is None:
                return "Unknown", 60.0
            if value >= 0.25:
                return "Wide", 90.0
            if value >= 0.15:
                return "Healthy", 75.0
            if value >= 0.05:
                return "Thin", 60.0
            return "Negative", 45.0

        gross_label, gross_score = classify_margin(gross_margin)
        op_label, op_score = classify_margin(operating_margin)
        net_label, net_score = classify_margin(net_margin)

        gross_history = margin_trend.get("gross_margins") or []
        op_history = margin_trend.get("operating_margins") or []
        net_history = margin_trend.get("net_margins") or []

        def classify_returns(value: Optional[float]) -> tuple:
            if value is None:
                return "Unknown", 60.0
            if value >= 0.18:
                return "High", 90.0
            if value >= 0.10:
                return "Solid", 75.0
            if value >= 0.05:
                return "Moderate", 60.0
            return "Low", 45.0

        roe_label, roe_score = classify_returns(roe)
        roa_label, roa_score = classify_returns(roa)

        margin_direction = margin_trend.get("net_margin_trend", "stable")
        direction_comment = {
            "expanding": "Margins trending higher.",
            "contracting": "Margins compressing.",
            "stable": "Margins steady year over year.",
        }.get(margin_direction, "Margin trend unavailable.")

        pricing_power_label = gross_label if gross_label != "Unknown" else op_label
        pricing_comment = f"Gross margin {pct(gross_margin)}; operating margin {pct(operating_margin)}."

        operating_leverage = None
        if gross_margin is not None and operating_margin is not None:
            operating_leverage = gross_margin - operating_margin
        if operating_leverage is None:
            leverage_label, leverage_score = "Unknown", 60.0
            leverage_comment = "Insufficient data to assess operating leverage."
        elif operating_leverage <= 0.05:
            leverage_label, leverage_score = "Low", 60.0
            leverage_comment = "Limited drop from gross to operating margin."
        elif operating_leverage <= 0.15:
            leverage_label, leverage_score = "Balanced", 75.0
            leverage_comment = "Moderate fixed-cost absorption."
        else:
            leverage_label, leverage_score = "High", 85.0
            leverage_comment = "High fixed-cost base amplifies swings."

        cost_structure_spread = None
        if gross_margin is not None and operating_margin is not None:
            cost_structure_spread = gross_margin - operating_margin
        if cost_structure_spread is None:
            cost_label, cost_score = "Unknown", 60.0
            cost_comment = "Unable to derive cost structure spread."
        elif cost_structure_spread <= 0.10:
            cost_label, cost_score = "Lean", 85.0
            cost_comment = "Operating expenses tightly managed."
        elif cost_structure_spread <= 0.20:
            cost_label, cost_score = "Balanced", 70.0
            cost_comment = "Cost structure consistent with peers."
        else:
            cost_label, cost_score = "Heavy", 55.0
            cost_comment = "Operating expenses absorb large share of gross profit."

        profitability_drivers: List[str] = []
        if gross_history:
            try:
                profitability_drivers.append(f"Median gross margin {statistics.median(gross_history):.1f}%")
            except statistics.StatisticsError:
                pass
        if net_history:
            try:
                median_net = statistics.median(net_history)
                if median_net > 15:
                    profitability_drivers.append("Consistent double-digit net margins")
            except statistics.StatisticsError:
                pass
        if gross_label in {"Wide", "Healthy"}:
            profitability_drivers.append("Premium margin profile")
        if roe_label in {"High", "Solid"}:
            profitability_drivers.append("Efficient capital deployment")
        if not profitability_drivers:
            profitability_drivers.append("Scaling opportunities")

        profitability_score = round(
            sum(
                [
                    gross_score,
                    op_score,
                    net_score,
                    roe_score,
                    roa_score,
                    leverage_score,
                    cost_score,
                ]
            )
            / 7,
            1,
        )

        payload = {
            "margin_trends_and_sustainability": {
                "assessment": margin_direction.capitalize(),
                "commentary": direction_comment,
            },
            "return_on_capital_efficiency": {
                "assessment": roe_label,
                "commentary": f"ROE {pct(roe)}; ROA {pct(roa)}.",
            },
            "competitive_advantages_moat": {
                "assessment": gross_label,
                "commentary": f"Gross margin {pct(gross_margin)} suggests {gross_label.lower()} moat.",
            },
            "pricing_power_indicators": {
                "assessment": pricing_power_label,
                "commentary": pricing_comment,
            },
            "cost_structure_analysis": {
                "assessment": cost_label,
                "commentary": cost_comment,
            },
            "operating_leverage": {
                "assessment": leverage_label,
                "commentary": leverage_comment,
            },
            "profitability_drivers": profitability_drivers,
            "profitability_score": profitability_score,
        }

        return self._build_deterministic_response("profitability_analysis", payload)


# Singleton instance
_analyzer_instance: Optional[DeterministicAnalyzer] = None


def get_deterministic_analyzer(
    agent_id: str = "deterministic", logger: Optional[logging.Logger] = None
) -> DeterministicAnalyzer:
    """
    Get singleton DeterministicAnalyzer instance.

    Args:
        agent_id: Agent identifier (only used on first call)
        logger: Optional logger (only used on first call)

    Returns:
        DeterministicAnalyzer instance
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = DeterministicAnalyzer(agent_id, logger)
    return _analyzer_instance
