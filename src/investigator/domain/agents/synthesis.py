"""
Synthesis Agent
Master agent that synthesizes insights from all specialized agents using Ollama LLMs
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from investigator.domain.agents.base import InvestmentAgent
from investigator.domain.models.analysis import AgentResult, AgentTask, TaskStatus
from investigator.domain.services.toon_formatter import to_toon_peers, to_toon_array, TOONFormatter
from investigator.domain.services.deterministic_conflict_resolver import reconcile_conflicts
from investigator.domain.services.deterministic_insight_extractor import extract_key_insights
from investigator.domain.services.template_thesis_generator import generate_investment_thesis
from investigator.infrastructure.cache import CacheManager


class AnalysisWeight(Enum):
    """Weight distribution for different analysis types"""

    SEC = 0.30
    FUNDAMENTAL = 0.35
    TECHNICAL = 0.20
    SENTIMENT = 0.15


@dataclass
class SynthesisInput:
    """Input data from various analysis agents"""

    symbol: str
    sec_analysis: Optional[Dict] = None
    fundamental_analysis: Optional[Dict] = None
    technical_analysis: Optional[Dict] = None
    sentiment_analysis: Optional[Dict] = None
    peer_comparison: Optional[Dict] = None
    market_context: Optional[Dict] = None
    context: Optional[Dict] = None
    timestamp: datetime = None


class SynthesisAgent(InvestmentAgent):
    """
    Master synthesis agent that combines insights from all specialized agents
    to produce comprehensive investment recommendations
    """

    def __init__(self, agent_id: str, ollama_client, event_bus, cache_manager: CacheManager):
        from investigator.config import get_config

        config = get_config()
        self.config = config  # Store config for TOON format access

        synthesis_model = config.ollama.models.get("synthesis", "deepseek-r1:32b")
        decision_model = config.ollama.models.get("decision", synthesis_model)
        summary_model = config.ollama.models.get("summary", synthesis_model)

        # Use the configured models for synthesis pipeline
        self.models = {
            "synthesis": synthesis_model,
            "reasoning": decision_model,
            "decision": decision_model,
            "summary": summary_model,
        }

        self.primary_model = synthesis_model
        super().__init__(agent_id, ollama_client, event_bus, cache_manager)

        # Decision thresholds
        self.thresholds = {"strong_buy": 80, "buy": 65, "hold": 50, "sell": 35, "strong_sell": 20}

        # Risk categories
        self.risk_categories = [
            "market_risk",
            "operational_risk",
            "financial_risk",
            "regulatory_risk",
            "competitive_risk",
            "execution_risk",
        ]

        # Deterministic processing config (replaces LLM calls with rule-based computation)
        valuation_config = getattr(config, "valuation", None)
        valuation_config_dict = valuation_config if isinstance(valuation_config, dict) else {}
        deterministic_config = valuation_config_dict.get("deterministic", {})
        self.use_deterministic = deterministic_config.get("enabled", True)
        self.deterministic_conflict_resolution = deterministic_config.get("conflict_resolution", True)
        self.deterministic_insight_extraction = deterministic_config.get("insight_extraction", True)
        self.deterministic_thesis_generation = deterministic_config.get("thesis_generation", True)

    def _build_deterministic_response(self, label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structure consistent with _wrap_llm_response for rule-based analyses."""
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

    def _parse_llm_response(self, response, default=None):
        """
        Safely parse LLM response that could be string or dict

        Args:
            response: LLM response (could be str, dict, or None)
            default: Default value to return if parsing fails (None or {})

        Returns:
            Parsed dict or default value
        """
        if default is None:
            default = {}

        # If already a dict, return as-is
        if isinstance(response, dict):
            return response

        # If None or empty, return default
        if not response:
            return default

        # If string, try to parse as JSON
        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
                self.logger.debug(f"Raw response: {response[:200]}...")
                return default

        # Unknown type, log warning and return default
        self.logger.warning(f"Unexpected LLM response type: {type(response)}")
        return default

    def register_capabilities(self) -> List:
        """Register agent capabilities"""
        from investigator.domain.agents.base import AgentCapability, AnalysisType

        return [
            AgentCapability(
                analysis_type=AnalysisType.INVESTMENT_SYNTHESIS,
                min_data_required={"analyses": dict},
                max_processing_time=120,  # Increased 2x for slower hardware
                required_models=[self.primary_model],
                cache_ttl=1800,
            )
        ]

    async def process(self, task: AgentTask) -> AgentResult:
        """Process synthesis task combining all analyses"""
        symbol = task.context.get("symbol")
        analyses = task.context.get("analyses", {})
        synthesis_type = task.context.get("synthesis_type", "comprehensive")

        self.logger.info(f"Synthesizing {synthesis_type} analysis for {symbol}")

        try:
            # CRITICAL: Normalize all input data to snake_case before processing
            # This ensures consistent key access across different agent outputs
            from investigator.domain.services.data_normalizer import DataNormalizer

            normalized_analyses = {}
            for analysis_type, analysis_data in analyses.items():
                if analysis_data and isinstance(analysis_data, dict):
                    try:
                        normalized_analyses[analysis_type] = DataNormalizer.normalize_and_round(
                            analysis_data, to_camel_case=False
                        )
                        self.logger.debug(
                            f"Normalized {analysis_type} analysis to snake_case " f"({len(analysis_data)} keys)"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to normalize {analysis_type} analysis: {e}. " f"Using original data."
                        )
                        normalized_analyses[analysis_type] = analysis_data
                else:
                    normalized_analyses[analysis_type] = analysis_data

            # Prepare synthesis input with normalized data
            synthesis_input = SynthesisInput(
                symbol=symbol,
                sec_analysis=normalized_analyses.get("sec"),
                fundamental_analysis=normalized_analyses.get("fundamental"),
                technical_analysis=normalized_analyses.get("technical"),
                sentiment_analysis=normalized_analyses.get("sentiment"),
                peer_comparison=normalized_analyses.get("peer_comparison"),
                market_context=normalized_analyses.get("market_context"),
                context=task.context,
                timestamp=datetime.now(),
            )

            # Validate and score individual analyses
            analysis_scores = await self._score_analyses(synthesis_input)

            # Identify key insights from each analysis
            key_insights = await self._extract_key_insights(synthesis_input)

            # Calculate multi-year CAGR and volatility metrics (Phase 1 enhancement)
            multi_year_trends = self._calculate_multi_year_metrics(synthesis_input)
            self.logger.info(
                f"Multi-year analysis: Revenue CAGR={multi_year_trends.get('revenue', {}).get('cagr', 'N/A')}%, "
                f"Pattern={multi_year_trends.get('revenue', {}).get('pattern', 'N/A')}, "
                f"Data Quality={multi_year_trends.get('data_quality', 'N/A')}"
            )

            # Generate charts for multi-year trends (Phase 2 enhancement)
            charts = self._generate_charts(symbol, synthesis_input, multi_year_trends)
            if charts["charts_generated"]:
                self.logger.info(
                    f"Charts generated: Quarterly={bool(charts['quarterly_revenue_chart'])}, "
                    f"Multi-year={bool(charts['multi_year_trends_chart'])}"
                )

            # Analyze comprehensive trends across revenue, margins, and cash flow (Phase 3 enhancement)
            trend_analysis = self._analyze_comprehensive_trends(synthesis_input)
            self.logger.info(
                f"Comprehensive trends: Revenue={trend_analysis.get('revenue_trend', 'N/A')}, "
                f"Margin={trend_analysis.get('margin_trend', 'N/A')}, "
                f"Cash Flow={trend_analysis.get('cash_flow_trend', 'N/A')}"
            )

            # Detect conflicts and reconcile
            conflicts = await self._detect_conflicts(synthesis_input)
            reconciliation = await self._reconcile_conflicts(conflicts, synthesis_input)

            # Calculate composite scores
            composite_scores = await self._calculate_composite_scores(synthesis_input, analysis_scores)

            # Generate investment thesis
            thesis = await self._generate_investment_thesis(synthesis_input, key_insights, composite_scores)

            # Assess risks comprehensively
            risk_assessment = await self._comprehensive_risk_assessment(synthesis_input, conflicts)

            # Generate scenarios
            scenarios = await self._generate_scenarios(synthesis_input, composite_scores, risk_assessment)

            # Make final recommendation
            recommendation = await self._make_recommendation(
                composite_scores, risk_assessment, scenarios, synthesis_input.symbol
            )

            # Generate action plan
            action_plan = await self._generate_action_plan(recommendation, synthesis_input, scenarios)

            # Create comprehensive report
            report = await self._create_synthesis_report(
                {
                    "symbol": symbol,
                    "timestamp": synthesis_input.timestamp.isoformat(),
                    "analysis_scores": analysis_scores,
                    "key_insights": key_insights,
                    "multi_year_trends": multi_year_trends,  # Phase 1: CAGR, volatility, patterns
                    "charts": charts,  # Phase 2: Chart paths
                    "trend_analysis": trend_analysis,  # Phase 3: Revenue, margin, cash flow trends
                    "conflicts": conflicts,
                    "reconciliation": reconciliation,
                    "composite_scores": composite_scores,
                    "investment_thesis": thesis,
                    "risk_assessment": risk_assessment,
                    "scenarios": scenarios,
                    "recommendation": recommendation,
                    "action_plan": action_plan,
                }
            )

            # Generate PDF report with embedded charts (Phase 4 enhancement)
            pdf_result = self._generate_pdf_report(symbol, report, charts, synthesis_input)
            if pdf_result["pdf_generated"]:
                self.logger.info(f"PDF report generated: {pdf_result['pdf_path']}")
            else:
                if pdf_result.get("error"):
                    self.logger.debug(f"PDF generation skipped: {pdf_result['error']}")

            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result_data={
                    "status": "success",
                    "symbol": symbol,
                    "analysis": report,  # FIX: Changed from 'synthesis' to 'analysis' for consistency with other agents
                    "synthesis": report,  # DEPRECATED: Keep for backward compatibility, will be removed in future
                    "recommendation": recommendation,
                    "confidence": composite_scores.get("confidence", 0),
                    "risk_score": risk_assessment.get("overall_risk", 50),
                    "action_plan": action_plan,
                    "pdf_report": pdf_result,  # Phase 4: PDF report path and status
                },
                processing_time=0,  # Will be calculated by base class
            )

        except Exception as e:
            self.logger.error(f"Synthesis failed for {symbol}: {e}")
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                result_data={"status": "error", "symbol": symbol, "error": str(e)},
                processing_time=0,
                error=str(e),
            )

    async def _score_analyses(self, synthesis_input: SynthesisInput) -> Dict:
        """Score the quality and completeness of each analysis"""
        scores = {}

        # Score SEC analysis
        if synthesis_input.sec_analysis:
            sec_score = self._evaluate_analysis_quality(
                synthesis_input.sec_analysis, required_fields=["metrics", "risks", "analysis"]
            )
            scores["sec"] = sec_score

        # Score fundamental analysis
        if synthesis_input.fundamental_analysis:
            fundamental_score = self._evaluate_analysis_quality(
                synthesis_input.fundamental_analysis, required_fields=["valuation", "quality_score", "analysis"]
            )
            scores["fundamental"] = fundamental_score

        # Score technical analysis
        if synthesis_input.technical_analysis:
            technical_score = self._evaluate_analysis_quality(
                synthesis_input.technical_analysis, required_fields=["signals", "levels", "analysis"]
            )
            scores["technical"] = technical_score

        # Score sentiment analysis
        if synthesis_input.sentiment_analysis:
            sentiment_score = self._evaluate_analysis_quality(
                synthesis_input.sentiment_analysis, required_fields=["sentiment_score", "trends"]
            )
            scores["sentiment"] = sentiment_score

        return scores

    async def _extract_key_insights(self, synthesis_input: SynthesisInput) -> Dict:
        """Extract key insights from each analysis using LLM"""
        # Check if deterministic insight extraction is enabled (saves tokens, faster)
        if self.use_deterministic and self.deterministic_insight_extraction:
            self.logger.debug(f"{synthesis_input.symbol} - Using deterministic insight extraction (LLM bypass)")

            response_data = extract_key_insights(
                fundamental=synthesis_input.fundamental_analysis,
                technical=synthesis_input.technical_analysis,
                sec=synthesis_input.sec_analysis,
                market_context=synthesis_input.market_context,
            )

            # Add quantitative insights (same as LLM path)
            response_data["quantitative"] = self._extract_quantitative_insights(synthesis_input)

            return self._build_deterministic_response("key_insights", response_data)

        # === LLM Path (fallback when deterministic is disabled) ===
        insights = {}

        # Prepare analysis summary for LLM
        analysis_summary = self._prepare_analysis_summary(synthesis_input)

        # Round numeric values to reduce token usage
        from investigator.domain.services.data_normalizer import DataNormalizer

        rounded_summary = DataNormalizer.round_financial_data(analysis_summary)

        summary_json = json.dumps(rounded_summary, indent=2)[:8000]
        schema_example = (
            "{\n"
            '  "sec": {\n'
            '    "positive_factors": ["..."],\n'
            '    "negative_factors": ["..."],\n'
            '    "critical_metric": "...",\n'
            '    "unique_insight": "...",\n'
            '    "confidence": 90\n'
            "  },\n"
            '  "fundamental": {\n'
            '    "positive_factors": ["..."],\n'
            '    "negative_factors": ["..."],\n'
            '    "critical_metric": "...",\n'
            '    "unique_insight": "...",\n'
            '    "confidence": 85\n'
            "  }\n"
            "}"
        )
        prompt = (
            "Extract the most important insights from each analysis type:\n\n"
            f"{summary_json}\n\n"
            "For each analysis type, identify:\n"
            "1. Top 3 positive factors\n"
            "2. Top 3 negative factors\n"
            "3. Most critical metric or finding\n"
            "4. Unique insight not found in other analyses\n"
            "5. Confidence level in the analysis\n\n"
            "Prioritize actionable, specific insights over generic observations.\n\n"
            "Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.\n\n"
            "Return a JSON object that strictly follows the schema below (values are illustrative):\n"
            f"{schema_example}\n"
        )

        response = await self.ollama.generate(
            model=self.models["reasoning"],
            prompt=prompt,
            format="json",
            prompt_name="_extract_key_insights_prompt",
        )

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["reasoning"],
            symbol=synthesis_input.symbol,
            llm_type="synthesis_extract_insights",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Parse response safely
        insights = self._parse_llm_response(response)

        # Add quantitative insights
        insights["quantitative"] = self._extract_quantitative_insights(synthesis_input)

        return self._wrap_llm_response(
            response=insights, model=self.models["reasoning"], prompt=prompt, temperature=0.3, top_p=0.9, format="json"
        )

    def _calculate_smart_price_targets(
        self, synthesis_input: SynthesisInput, composite_scores: Dict, risk_assessment: Dict
    ) -> Dict:
        """Calculate smart price targets with both upward and downward adjustments"""
        from utils.valuation.valuation_adjustments import SmartValuationAdjuster, ValuationMetrics

        # Extract valuation data
        valuation_data = (
            synthesis_input.fundamental_analysis.get("valuation", {}) if synthesis_input.fundamental_analysis else {}
        )
        technical_data = (
            synthesis_input.technical_analysis.get("signals", {}) if synthesis_input.technical_analysis else {}
        )

        # Extract multi-model summary for blended fair value
        multi_model_summary = (
            synthesis_input.fundamental_analysis.get("multi_model_summary", {})
            if synthesis_input.fundamental_analysis
            else {}
        )
        blended_fair_value = multi_model_summary.get("blended_fair_value")
        model_agreement_score = multi_model_summary.get("model_agreement_score")
        overall_confidence = multi_model_summary.get("overall_confidence")

        # Use blended fair value as primary (fallback to regular fair_value if unavailable)
        final_fair_value = blended_fair_value if blended_fair_value else valuation_data.get("fair_value", 0)

        # Create valuation metrics
        metrics = ValuationMetrics(
            current_price=valuation_data.get("current_price", 0),
            fair_value=final_fair_value,  # Use blended or fallback
            technical_target=technical_data.get("target_price", 0),
            pe_ratio=valuation_data.get("pe_ratio"),
            pb_ratio=valuation_data.get("pb_ratio"),
            ps_ratio=valuation_data.get("ps_ratio"),
            quality_score=(
                synthesis_input.fundamental_analysis.get("quality_score", 50)
                if synthesis_input.fundamental_analysis
                else 50
            ),
            dividend_yield=valuation_data.get("dividend_yield"),
        )

        # Get market context if available
        market_context = synthesis_input.market_context if hasattr(synthesis_input, "market_context") else None

        # Prepare analysis context
        analysis_context = {
            "risk_level": self._map_risk_score_to_level(risk_assessment.get("overall_risk", 50)),
            "technical_trend": self._extract_technical_trend(synthesis_input.technical_analysis),
            "market_sentiment": self._assess_market_sentiment(synthesis_input, market_context),
            "sector": synthesis_input.context.get("sector", "default"),
            "quality_factors": self._extract_quality_factors(synthesis_input.fundamental_analysis),
            "market_regime": (
                market_context.get("market_sentiment", {}).get("market_regime") if market_context else "neutral"
            ),
            "sector_strength": (
                market_context.get("sector_context", {}).get("sector_strength") if market_context else "neutral"
            ),
        }

        # Calculate smart adjusted target
        adjuster = SmartValuationAdjuster()
        adjusted_target, adjustment_details = adjuster.calculate_adjusted_target(metrics, analysis_context)

        # Generate valuation summary
        valuation_summary = adjuster.generate_valuation_summary(metrics, adjusted_target, adjustment_details)

        return {
            "adjusted_target": adjusted_target,
            "adjustment_details": adjustment_details,
            "valuation_summary": valuation_summary,
            "base_metrics": {
                "current_price": metrics.current_price,
                "fair_value": metrics.fair_value,
                "technical_target": metrics.technical_target,
                "valuation_bias": metrics.valuation_bias.value,
                "quality_tier": metrics.quality_tier.value,
            },
        }

    def _map_risk_score_to_level(self, risk_score: float) -> str:
        """Map risk score to risk level"""
        if risk_score >= 80:
            return "very_high"
        elif risk_score >= 60:
            return "high"
        elif risk_score >= 40:
            return "medium"
        elif risk_score >= 20:
            return "low"
        else:
            return "very_low"

    def _extract_technical_trend(self, technical_analysis: Optional[Dict]) -> str:
        """Extract technical trend from analysis"""
        if not technical_analysis:
            return "sideways"

        analysis = technical_analysis.get("analysis", {})
        trend = analysis.get("trend", "").lower()

        if "strong" in trend and "up" in trend:
            return "strong_uptrend"
        elif "up" in trend or "bullish" in trend:
            return "uptrend"
        elif "down" in trend or "bearish" in trend:
            return "downtrend"
        elif "strong" in trend and "down" in trend:
            return "strong_downtrend"
        else:
            return "sideways"

    def _assess_market_sentiment(self, synthesis_input: SynthesisInput, market_context: Optional[Dict] = None) -> str:
        """Assess overall market sentiment using ETF context data"""
        # If we have ETF market context, use it for better sentiment assessment
        if market_context:
            market_sentiment = market_context.get("market_sentiment", {})

            # Extract sentiment from LLM analysis
            if "sentiment" in market_sentiment:
                return market_sentiment["sentiment"]

            # Extract from market regime
            market_regime = market_sentiment.get("market_regime", "neutral")
            if market_regime == "risk_on":
                return "bullish"
            elif market_regime == "risk_off":
                return "bearish"

            # Fall back to market performance
            market_perf = market_context.get("market_context", {}).get("medium_term", {})
            spy_data = market_perf.get("broad_market", {})
            spy_return = spy_data.get("return", 0)

            if spy_return > 0.10:  # >10% return
                return "very_bullish"
            elif spy_return > 0.05:  # >5% return
                return "bullish"
            elif spy_return > -0.05:  # -5% to +5%
                return "neutral"
            elif spy_return > -0.10:  # -10% to -5%
                return "bearish"
            else:  # <-10%
                return "very_bearish"

        # Fallback to original logic if no market context
        sentiment_score = 50  # Default neutral

        if synthesis_input.fundamental_analysis:
            fund_score = synthesis_input.fundamental_analysis.get("quality_score", 50)
            sentiment_score = (sentiment_score + fund_score) / 2

        if sentiment_score >= 80:
            return "very_bullish"
        elif sentiment_score >= 65:
            return "bullish"
        elif sentiment_score >= 35:
            return "neutral"
        elif sentiment_score >= 20:
            return "bearish"
        else:
            return "very_bearish"

    def _extract_quality_factors(self, fundamental_analysis: Optional[Dict]) -> Dict:
        """Extract quality factors from fundamental analysis"""
        if not fundamental_analysis:
            return {}

        return {
            "quality_score": fundamental_analysis.get("quality_score", 50),
            "competitive_advantages": fundamental_analysis.get("competitive_advantages", []),
            "management_quality": fundamental_analysis.get("management_quality", "average"),
            "financial_strength": fundamental_analysis.get("financial_strength", "average"),
        }

    def _aggregate_to_fiscal_years(self, quarterly_data: List[Dict], metric_name: str) -> List[Dict]:
        """
        Aggregate quarterly financial data into fiscal years

        Args:
            quarterly_data: List of quarterly data points with 'period' and metric values
            metric_name: Name of the metric to aggregate (e.g., 'revenue', 'net_income')

        Returns:
            List of fiscal year aggregates [{'fiscal_year': 2023, 'value': 123.45}, ...]
        """
        if not quarterly_data:
            return []

        # Group quarterly data by fiscal year
        fiscal_years = {}
        for quarter in quarterly_data:
            # Extract fiscal year from period (e.g., "2023Q4" -> 2023)
            period = quarter.get("period", "")
            if not period:
                continue

            try:
                # Handle different period formats: "2023Q4", "FY2023Q4", "2023-Q4"
                year_str = period.replace("FY", "").replace("-", "").split("Q")[0]
                fiscal_year = int(year_str)

                # Get metric value
                value = quarter.get(metric_name)
                if value is None:
                    continue

                # Initialize fiscal year if not exists
                if fiscal_year not in fiscal_years:
                    fiscal_years[fiscal_year] = {"total": 0, "count": 0}

                # Accumulate value
                fiscal_years[fiscal_year]["total"] += float(value)
                fiscal_years[fiscal_year]["count"] += 1

            except (ValueError, IndexError, AttributeError) as e:
                self.logger.debug(f"Failed to parse period {period}: {e}")
                continue

        # Convert to list and sort by year (descending)
        annual_data = []
        for year, data in sorted(fiscal_years.items(), reverse=True):
            # Only include fiscal years with all 4 quarters (complete year)
            if data["count"] >= 4:
                annual_data.append({"fiscal_year": year, "value": data["total"]})

        return annual_data

    def _detect_cyclical_pattern(self, cagr: float, volatility: float) -> str:
        """
        Detect cyclical business pattern based on CAGR and volatility

        Args:
            cagr: Compound Annual Growth Rate (percentage, e.g., 15.3)
            volatility: Revenue/Earnings volatility (percentage, e.g., 8.2)

        Returns:
            Pattern classification: stable_growth, high_growth, volatile_growth,
                                   declining, cyclical, mature
        """
        # Classification thresholds (from old synthesizer.py logic)
        HIGH_GROWTH_THRESHOLD = 15.0  # 15%+ CAGR
        STABLE_GROWTH_THRESHOLD = 5.0  # 5%+ CAGR
        LOW_VOLATILITY_THRESHOLD = 10.0  # <10% volatility = predictable
        HIGH_VOLATILITY_THRESHOLD = 25.0  # >25% volatility = high risk

        # Classify based on growth and volatility
        if cagr >= HIGH_GROWTH_THRESHOLD:
            if volatility < LOW_VOLATILITY_THRESHOLD:
                return "stable_growth"  # High growth, low volatility = IDEAL
            elif volatility < HIGH_VOLATILITY_THRESHOLD:
                return "high_growth"  # High growth, moderate volatility = GOOD
            else:
                return "volatile_growth"  # High growth but unpredictable = RISKY

        elif cagr >= STABLE_GROWTH_THRESHOLD:
            if volatility < LOW_VOLATILITY_THRESHOLD:
                return "mature"  # Moderate growth, low volatility = STABLE
            elif volatility < HIGH_VOLATILITY_THRESHOLD:
                return "cyclical"  # Moderate growth, moderate volatility
            else:
                return "volatile_growth"  # Moderate growth, high volatility = RISKY

        elif cagr >= 0:
            if volatility < LOW_VOLATILITY_THRESHOLD:
                return "mature"  # Low growth, low volatility = MATURE
            else:
                return "cyclical"  # Low growth, high volatility = CYCLICAL

        else:  # Negative CAGR
            return "declining"  # Negative growth = AVOID

    def _detect_trend_direction(self, annual_data: List[Dict]) -> str:
        """
        Detect trend direction using linear regression on annual data

        Args:
            annual_data: List of annual data points [{'fiscal_year': 2023, 'value': 123.45}, ...]

        Returns:
            Trend direction: 'UPWARD', 'DOWNWARD', or 'STABLE'
        """
        if not annual_data or len(annual_data) < 2:
            return "STABLE"

        # Extract years and values for regression
        years = [item["fiscal_year"] for item in annual_data]
        values = [item["value"] for item in annual_data]

        # Simple linear regression: y = mx + b
        # Calculate slope (m)
        n = len(years)
        mean_x = np.mean(years)
        mean_y = np.mean(values)

        numerator = sum((years[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((years[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return "STABLE"

        slope = numerator / denominator

        # Determine trend based on slope
        # Slope is in units per year, normalize by mean to get percentage change
        if mean_y != 0:
            slope_percent = (slope / mean_y) * 100  # Percentage change per year
        else:
            slope_percent = 0

        # Thresholds for trend classification
        UPWARD_THRESHOLD = 2.0  # >2% increase per year
        DOWNWARD_THRESHOLD = -2.0  # >2% decrease per year

        if slope_percent >= UPWARD_THRESHOLD:
            return "UPWARD"
        elif slope_percent <= DOWNWARD_THRESHOLD:
            return "DOWNWARD"
        else:
            return "STABLE"

    def _calculate_multi_year_metrics(self, synthesis_input: SynthesisInput) -> Dict:
        """
        Calculate multi-year CAGR, volatility, and trend metrics

        Args:
            synthesis_input: Synthesis input containing fundamental analysis data

        Returns:
            Dict with multi-year metrics:
            {
                'revenue': {
                    'cagr': 15.3,  # 5-year CAGR
                    'volatility': 8.2,  # Standard deviation %
                    'trend': 'UPWARD',
                    'pattern': 'stable_growth'
                },
                'earnings': {...},
                'data_quality': 'excellent' | 'good' | 'limited' | 'insufficient'
            }
        """
        metrics = {"revenue": {}, "earnings": {}, "data_quality": "insufficient", "years_analyzed": 0}

        # Check if fundamental analysis is available
        if not synthesis_input.fundamental_analysis:
            self.logger.debug("No fundamental analysis available for multi-year metrics")
            return metrics

        fundamental_data = synthesis_input.fundamental_analysis

        # Extract quarterly financial data (look for common keys)
        quarterly_data = fundamental_data.get("quarterly_metrics", [])
        if not quarterly_data:
            quarterly_data = fundamental_data.get("financials", {}).get("quarterly", [])
        if not quarterly_data:
            quarterly_data = fundamental_data.get("quarterly", [])

        if not quarterly_data:
            self.logger.debug("No quarterly data found in fundamental analysis")
            return metrics

        # --- REVENUE METRICS ---
        revenue_annual = self._aggregate_to_fiscal_years(quarterly_data, "revenue")

        if len(revenue_annual) >= 2:
            # Calculate Revenue CAGR
            latest_year = revenue_annual[0]
            oldest_year = revenue_annual[-1]
            years_diff = latest_year["fiscal_year"] - oldest_year["fiscal_year"]

            if years_diff > 0 and oldest_year["value"] > 0:
                revenue_cagr = (((latest_year["value"] / oldest_year["value"]) ** (1 / years_diff)) - 1) * 100
                metrics["revenue"]["cagr"] = round(revenue_cagr, 2)

                # Calculate Revenue Volatility (year-over-year growth rate standard deviation)
                growth_rates = []
                for i in range(len(revenue_annual) - 1):
                    current = revenue_annual[i]["value"]
                    previous = revenue_annual[i + 1]["value"]
                    if previous > 0:
                        yoy_growth = ((current - previous) / previous) * 100
                        growth_rates.append(yoy_growth)

                if growth_rates:
                    revenue_volatility = np.std(growth_rates)
                    metrics["revenue"]["volatility"] = round(revenue_volatility, 2)

                    # Detect pattern and trend
                    metrics["revenue"]["pattern"] = self._detect_cyclical_pattern(revenue_cagr, revenue_volatility)
                    metrics["revenue"]["trend"] = self._detect_trend_direction(revenue_annual)

        # --- EARNINGS METRICS ---
        earnings_annual = self._aggregate_to_fiscal_years(quarterly_data, "net_income")
        if not earnings_annual:
            # Try alternative field names
            earnings_annual = self._aggregate_to_fiscal_years(quarterly_data, "earnings")

        if len(earnings_annual) >= 2:
            # Calculate Earnings CAGR
            latest_year = earnings_annual[0]
            oldest_year = earnings_annual[-1]
            years_diff = latest_year["fiscal_year"] - oldest_year["fiscal_year"]

            if years_diff > 0 and oldest_year["value"] > 0:
                earnings_cagr = (((latest_year["value"] / oldest_year["value"]) ** (1 / years_diff)) - 1) * 100
                metrics["earnings"]["cagr"] = round(earnings_cagr, 2)

                # Calculate Earnings Volatility
                growth_rates = []
                for i in range(len(earnings_annual) - 1):
                    current = earnings_annual[i]["value"]
                    previous = earnings_annual[i + 1]["value"]
                    if previous > 0:
                        yoy_growth = ((current - previous) / previous) * 100
                        growth_rates.append(yoy_growth)

                if growth_rates:
                    earnings_volatility = np.std(growth_rates)
                    metrics["earnings"]["volatility"] = round(earnings_volatility, 2)

                    # Detect pattern and trend
                    metrics["earnings"]["pattern"] = self._detect_cyclical_pattern(earnings_cagr, earnings_volatility)
                    metrics["earnings"]["trend"] = self._detect_trend_direction(earnings_annual)

        # --- DATA QUALITY ASSESSMENT ---
        max_years = max(len(revenue_annual), len(earnings_annual))
        metrics["years_analyzed"] = max_years

        if max_years >= 5:
            metrics["data_quality"] = "excellent"
        elif max_years >= 3:
            metrics["data_quality"] = "good"
        elif max_years >= 2:
            metrics["data_quality"] = "limited"
        else:
            metrics["data_quality"] = "insufficient"

        return metrics

    def _generate_charts(self, symbol: str, synthesis_input: SynthesisInput, multi_year_metrics: Dict) -> Dict:
        """
        Generate charts for quarterly revenue trends and multi-year historical analysis

        Args:
            symbol: Stock symbol
            synthesis_input: Synthesis input containing fundamental analysis data
            multi_year_metrics: Multi-year metrics from _calculate_multi_year_metrics()

        Returns:
            Dict with chart paths:
            {
                'quarterly_revenue_chart': '/path/to/chart.png',
                'multi_year_trends_chart': '/path/to/chart.png',
                'charts_generated': True/False
            }
        """
        charts = {"quarterly_revenue_chart": "", "multi_year_trends_chart": "", "charts_generated": False}

        try:
            # Import ChartGenerator
            from pathlib import Path

            from utils.chart_generator import ChartGenerator

            # Check if fundamental analysis data is available
            if not synthesis_input.fundamental_analysis:
                self.logger.debug("No fundamental analysis available for chart generation")
                return charts

            fundamental_data = synthesis_input.fundamental_analysis

            # Create chart generator with output directory
            charts_dir = Path("charts") / symbol
            chart_gen = ChartGenerator(charts_dir)

            # Extract quarterly data for revenue trend chart
            quarterly_data = fundamental_data.get("quarterly_metrics", [])
            if not quarterly_data:
                quarterly_data = fundamental_data.get("financials", {}).get("quarterly", [])
            if not quarterly_data:
                quarterly_data = fundamental_data.get("quarterly", [])

            # Generate quarterly revenue trend chart
            if quarterly_data:
                # Prepare data in expected format
                revenue_trend_data = []
                for quarter in quarterly_data:
                    period = quarter.get("period", "")
                    revenue = quarter.get("revenue", 0)
                    if period and revenue:
                        revenue_trend_data.append({"period": period, "value": revenue})

                if revenue_trend_data:
                    quarterly_trends = {"revenue_trend": revenue_trend_data}
                    quarterly_chart_path = chart_gen.generate_quarterly_revenue_trend(symbol, quarterly_trends)
                    if quarterly_chart_path:
                        charts["quarterly_revenue_chart"] = quarterly_chart_path
                        self.logger.info(f"Generated quarterly revenue chart: {quarterly_chart_path}")

            # Generate multi-year trends chart
            if multi_year_metrics.get("data_quality") in ["excellent", "good", "limited"]:
                # Prepare yearly data from quarterly data
                annual_data = []

                # Get revenue and earnings annual data
                revenue_annual = self._aggregate_to_fiscal_years(quarterly_data, "revenue")
                earnings_annual = self._aggregate_to_fiscal_years(quarterly_data, "net_income")

                # Combine revenue and earnings by fiscal year
                years_revenue = {item["fiscal_year"]: item["value"] for item in revenue_annual}
                years_earnings = {item["fiscal_year"]: item["value"] for item in earnings_annual}

                # Get all years
                all_years = sorted(set(years_revenue.keys()) | set(years_earnings.keys()), reverse=True)

                for year in all_years:
                    annual_data.append(
                        {
                            "fiscal_year": year,
                            "revenue": years_revenue.get(year, 0),
                            "net_income": years_earnings.get(year, 0),
                        }
                    )

                # Prepare metrics in expected format
                metrics = {
                    "revenue_cagr": multi_year_metrics.get("revenue", {}).get("cagr"),
                    "cyclical_pattern": multi_year_metrics.get("revenue", {}).get("pattern"),
                }

                if annual_data and len(annual_data) >= 2:
                    multi_year_trends_data = {"data": annual_data, "metrics": metrics}
                    multi_year_chart_path = chart_gen.generate_multi_year_trends_chart(symbol, multi_year_trends_data)
                    if multi_year_chart_path:
                        charts["multi_year_trends_chart"] = multi_year_chart_path
                        self.logger.info(f"Generated multi-year trends chart: {multi_year_chart_path}")

            # Set charts_generated flag if at least one chart was created
            charts["charts_generated"] = bool(charts["quarterly_revenue_chart"] or charts["multi_year_trends_chart"])

        except ImportError as e:
            self.logger.warning(f"Chart generation not available: {e}")
        except Exception as e:
            self.logger.error(f"Failed to generate charts for {symbol}: {e}")

        return charts

    def _classify_revenue_trend(self, growth_rates: List[float]) -> str:
        """
        Classify revenue trend based on growth rate progression

        Args:
            growth_rates: List of YoY growth rates (most recent first)

        Returns:
            'accelerating_growth', 'stable_growth', 'decelerating_growth', or 'declining'

        Example:
            growth_rates = [25.0, 20.0, 15.0] -> 'accelerating_growth'
            growth_rates = [15.0, 14.5, 15.2] -> 'stable_growth'
            growth_rates = [10.0, 15.0, 20.0] -> 'decelerating_growth'
            growth_rates = [-5.0, -3.0, -2.0] -> 'declining'
        """
        if not growth_rates or len(growth_rates) < 2:
            return "insufficient_data"

        # Check if declining (negative growth)
        avg_growth = np.mean(growth_rates)
        if avg_growth < 0:
            return "declining"

        # Calculate acceleration (change in growth rate)
        acceleration = []
        for i in range(len(growth_rates) - 1):
            accel = growth_rates[i] - growth_rates[i + 1]
            acceleration.append(accel)

        avg_acceleration = np.mean(acceleration)

        # Thresholds for classification
        ACCELERATING_THRESHOLD = 2.0  # Growth rate improving by >2% per period
        DECELERATING_THRESHOLD = -2.0  # Growth rate declining by >2% per period

        if avg_acceleration >= ACCELERATING_THRESHOLD:
            return "accelerating_growth"
        elif avg_acceleration <= DECELERATING_THRESHOLD:
            return "decelerating_growth"
        else:
            return "stable_growth"

    def _classify_margin_trend(self, margins: List[float]) -> str:
        """
        Classify margin trend based on margin progression

        Args:
            margins: List of margin percentages (most recent first)

        Returns:
            'expanding', 'stable', or 'contracting'

        Example:
            margins = [12.5, 12.0, 11.5] -> 'expanding'
            margins = [12.0, 12.1, 11.9] -> 'stable'
            margins = [11.0, 11.5, 12.0] -> 'contracting'
        """
        if not margins or len(margins) < 2:
            return "insufficient_data"

        # Calculate margin change trend using linear regression
        n = len(margins)
        x = list(range(n))  # Time index
        y = margins

        mean_x = np.mean(x)
        mean_y = np.mean(y)

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Thresholds (basis points per period)
        EXPANDING_THRESHOLD = 0.5  # Margin improving by >0.5% per period
        CONTRACTING_THRESHOLD = -0.5  # Margin declining by >0.5% per period

        if slope >= EXPANDING_THRESHOLD:
            return "expanding"
        elif slope <= CONTRACTING_THRESHOLD:
            return "contracting"
        else:
            return "stable"

    def _classify_cash_flow_trend(self, cash_flows: List[float]) -> str:
        """
        Classify cash flow trend based on cash flow progression

        Args:
            cash_flows: List of free cash flow values (most recent first)

        Returns:
            'improving', 'stable', or 'deteriorating'

        Example:
            cash_flows = [1500, 1200, 1000] -> 'improving'
            cash_flows = [1000, 1020, 980] -> 'stable'
            cash_flows = [800, 1000, 1200] -> 'deteriorating'
        """
        if not cash_flows or len(cash_flows) < 2:
            return "insufficient_data"

        # Calculate YoY growth rates
        growth_rates = []
        for i in range(len(cash_flows) - 1):
            current = cash_flows[i]
            previous = cash_flows[i + 1]
            if previous != 0:
                growth = ((current - previous) / abs(previous)) * 100
                growth_rates.append(growth)

        if not growth_rates:
            return "insufficient_data"

        avg_growth = np.mean(growth_rates)

        # Thresholds
        IMPROVING_THRESHOLD = 10.0  # Cash flow improving by >10% on average
        DETERIORATING_THRESHOLD = -10.0  # Cash flow declining by >10% on average

        if avg_growth >= IMPROVING_THRESHOLD:
            return "improving"
        elif avg_growth <= DETERIORATING_THRESHOLD:
            return "deteriorating"
        else:
            return "stable"

    def _analyze_comprehensive_trends(self, synthesis_input: SynthesisInput) -> Dict:
        """
        Analyze comprehensive trends across revenue, margins, and cash flow

        This method calculates quarter-over-quarter (QoQ) and year-over-year (YoY) trends
        to detect business momentum and quality changes.

        Args:
            synthesis_input: SynthesisInput with fundamental analysis data

        Returns:
            Dictionary with trend analysis:
            {
                'revenue_trend': 'accelerating_growth|stable_growth|decelerating_growth|declining',
                'margin_trend': 'expanding|stable|contracting',
                'cash_flow_trend': 'improving|stable|deteriorating',
                'quarterly_insights': {
                    'recent_quarters_analyzed': int,
                    'revenue_momentum': 'positive|neutral|negative',
                    'margin_quality': 'improving|stable|deteriorating',
                    'cash_generation': 'strong|adequate|weak'
                }
            }

        Example Output:
            {
                'revenue_trend': 'accelerating_growth',
                'margin_trend': 'expanding',
                'cash_flow_trend': 'improving',
                'quarterly_insights': {
                    'recent_quarters_analyzed': 8,
                    'revenue_momentum': 'positive',
                    'margin_quality': 'improving',
                    'cash_generation': 'strong'
                }
            }
        """
        trend_analysis = {
            "revenue_trend": "insufficient_data",
            "margin_trend": "insufficient_data",
            "cash_flow_trend": "insufficient_data",
            "quarterly_insights": {
                "recent_quarters_analyzed": 0,
                "revenue_momentum": "neutral",
                "margin_quality": "neutral",
                "cash_generation": "neutral",
            },
        }

        if not synthesis_input.fundamental_analysis:
            return trend_analysis

        fundamental_data = synthesis_input.fundamental_analysis

        # Extract quarterly data (try multiple locations)
        quarterly_data = fundamental_data.get("quarterly_metrics", [])
        if not quarterly_data:
            quarterly_data = fundamental_data.get("financials", {}).get("quarterly", [])
        if not quarterly_data:
            quarterly_data = fundamental_data.get("quarterly", [])

        if not quarterly_data or len(quarterly_data) < 4:
            self.logger.debug("Insufficient quarterly data for comprehensive trend analysis")
            return trend_analysis

        # Sort by period (most recent first)
        sorted_data = sorted(quarterly_data, key=lambda x: x.get("period", ""), reverse=True)

        # Limit to last 8 quarters (2 years)
        recent_data = sorted_data[:8]
        trend_analysis["quarterly_insights"]["recent_quarters_analyzed"] = len(recent_data)

        # Extract revenue data and calculate YoY growth
        revenue_values = []
        revenue_growth_rates = []
        for i, quarter in enumerate(recent_data):
            revenue = quarter.get("revenue", 0)
            if revenue:
                revenue_values.append(revenue)

                # Calculate YoY growth (compare to 4 quarters ago)
                if i + 4 < len(recent_data):
                    prev_year_quarter = recent_data[i + 4]
                    prev_revenue = prev_year_quarter.get("revenue", 0)
                    if prev_revenue > 0:
                        yoy_growth = ((revenue - prev_revenue) / prev_revenue) * 100
                        revenue_growth_rates.append(yoy_growth)

        # Analyze revenue trend
        if revenue_growth_rates:
            trend_analysis["revenue_trend"] = self._classify_revenue_trend(revenue_growth_rates)

            # Determine revenue momentum
            avg_growth = np.mean(revenue_growth_rates)
            if avg_growth >= 10.0:
                trend_analysis["quarterly_insights"]["revenue_momentum"] = "positive"
            elif avg_growth <= -5.0:
                trend_analysis["quarterly_insights"]["revenue_momentum"] = "negative"

        # Extract margin data (operating margin or net margin)
        margin_values = []
        for quarter in recent_data:
            # Try to get operating margin first, then net margin
            margin = quarter.get("operating_margin")
            if margin is None:
                # Calculate from operating income / revenue
                operating_income = quarter.get("operating_income", 0)
                revenue = quarter.get("revenue", 0)
                if revenue > 0:
                    margin = (operating_income / revenue) * 100

            if margin is not None and margin != 0:
                margin_values.append(margin)

        # Analyze margin trend
        if len(margin_values) >= 4:
            trend_analysis["margin_trend"] = self._classify_margin_trend(margin_values)

            # Determine margin quality
            if trend_analysis["margin_trend"] == "expanding":
                trend_analysis["quarterly_insights"]["margin_quality"] = "improving"
            elif trend_analysis["margin_trend"] == "contracting":
                trend_analysis["quarterly_insights"]["margin_quality"] = "deteriorating"

        # Extract cash flow data
        cash_flow_values = []
        for quarter in recent_data:
            # Try free cash flow first, then operating cash flow
            fcf = quarter.get("free_cash_flow")
            if fcf is None:
                fcf = quarter.get("operating_cash_flow", 0)

            if fcf is not None:
                cash_flow_values.append(fcf)

        # Analyze cash flow trend
        if len(cash_flow_values) >= 4:
            trend_analysis["cash_flow_trend"] = self._classify_cash_flow_trend(cash_flow_values)

            # Determine cash generation quality
            avg_cash_flow = np.mean(cash_flow_values)
            if trend_analysis["cash_flow_trend"] == "improving" and avg_cash_flow > 0:
                trend_analysis["quarterly_insights"]["cash_generation"] = "strong"
            elif trend_analysis["cash_flow_trend"] == "deteriorating" or avg_cash_flow < 0:
                trend_analysis["quarterly_insights"]["cash_generation"] = "weak"
            else:
                trend_analysis["quarterly_insights"]["cash_generation"] = "adequate"

        self.logger.info(
            f"Comprehensive trend analysis: Revenue={trend_analysis['revenue_trend']}, "
            f"Margin={trend_analysis['margin_trend']}, "
            f"Cash Flow={trend_analysis['cash_flow_trend']}"
        )

        return trend_analysis

    def _generate_pdf_report(
        self, symbol: str, synthesis_report: Dict, chart_paths: Dict, synthesis_input: "SynthesisInput"
    ) -> Dict:
        """
        Generate professional PDF report with embedded charts

        This method integrates with the existing PDFReportGenerator to create
        investor-ready PDF reports with all synthesis data and visualizations.

        Args:
            symbol: Stock symbol
            synthesis_report: Complete synthesis report data
            chart_paths: Dictionary with chart file paths from Phase 2

        Returns:
            Dictionary with PDF generation result:
            {
                'pdf_path': str (path to generated PDF),
                'pdf_generated': bool,
                'error': str (if failed)
            }

        Example Output:
            {
                'pdf_path': 'reports/AAPL_synthesis_2025-10-30.pdf',
                'pdf_generated': True
            }
        """
        result = {"pdf_path": "", "pdf_generated": False, "error": None}

        try:
            from pathlib import Path

            from investigator.infrastructure.reporting import (
                PDFReportGenerator,
                ReportConfig,
                ReportPayloadBuilder,
            )

            # Check if report generator available
            try:
                from reportlab.lib.pagesizes import letter

                reportlab_available = True
            except ImportError:
                self.logger.warning("reportlab not available - PDF generation disabled")
                result["error"] = "reportlab_not_installed"
                return result

            # Create output directory for reports
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            # Initialize PDF generator with default config
            config = ReportConfig()
            pdf_gen = PDFReportGenerator(reports_dir, config)

            # Use ReportPayloadBuilder to create normalized PDF payload
            # This handles all data transformations, unwrapping, score scaling, and validation
            payload_builder = ReportPayloadBuilder(logger=self.logger)

            # Collect chart paths
            include_charts = []
            if chart_paths.get("charts_generated"):
                if chart_paths.get("quarterly_revenue_chart"):
                    include_charts.append(chart_paths["quarterly_revenue_chart"])
                if chart_paths.get("multi_year_trends_chart"):
                    include_charts.append(chart_paths["multi_year_trends_chart"])

            # Get fundamental and technical data from synthesis input for backfilling
            fundamental_data = (
                synthesis_input.fundamental_analysis if hasattr(synthesis_input, "fundamental_analysis") else None
            )
            technical_data = (
                synthesis_input.technical_analysis if hasattr(synthesis_input, "technical_analysis") else None
            )

            # Build normalized recommendation payload
            recommendation = payload_builder.build(
                symbol=symbol,
                synthesis_report=synthesis_report,
                fundamental_data=fundamental_data,
                technical_data=technical_data,
                chart_paths=include_charts,
            )

            # Generate PDF report
            self.logger.info(f"Generating PDF report for {symbol} with {len(include_charts)} charts")

            pdf_path = pdf_gen.generate_report(
                recommendations=[recommendation],
                report_type="synthesis",
                include_charts=include_charts if include_charts else None,
            )

            if pdf_path:
                result["pdf_path"] = pdf_path
                result["pdf_generated"] = True
                self.logger.info(f"PDF report generated successfully: {pdf_path}")
            else:
                result["error"] = "pdf_generation_failed"
                self.logger.warning("PDF generation returned empty path")

        except ImportError as e:
            self.logger.warning(f"PDF generation dependencies not available: {e}")
            result["error"] = f"import_error: {str(e)}"
        except Exception as e:
            self.logger.error(f"Failed to generate PDF report for {symbol}: {e}")
            result["error"] = f"generation_error: {str(e)}"

        return result

    async def _detect_conflicts(self, synthesis_input: SynthesisInput) -> List[Dict]:
        """Detect conflicts between different analyses"""
        conflicts = []

        # Check fundamental vs technical conflicts
        if synthesis_input.fundamental_analysis and synthesis_input.technical_analysis:
            fund_rec = synthesis_input.fundamental_analysis.get("recommendation", "hold")
            tech_rec = synthesis_input.technical_analysis.get("recommendation", "hold")

            if self._are_recommendations_conflicting(fund_rec, tech_rec):
                conflicts.append(
                    {
                        "type": "recommendation_conflict",
                        "analyses": ["fundamental", "technical"],
                        "fundamental_view": fund_rec,
                        "technical_view": tech_rec,
                        "severity": "high",
                    }
                )

        # Check valuation conflicts
        if synthesis_input.fundamental_analysis:
            valuation = synthesis_input.fundamental_analysis.get("valuation", {})
            # Use multi-model blended fair value if available
            multi_model_summary = synthesis_input.fundamental_analysis.get("multi_model_summary", {})
            blended_fair_value = multi_model_summary.get("blended_fair_value")
            fair_value = blended_fair_value if blended_fair_value else valuation.get("fair_value", 0)
            current_price = valuation.get("current_price", 0)

            if synthesis_input.technical_analysis:
                tech_signals = synthesis_input.technical_analysis.get("signals", {})
                tech_target = tech_signals.get("target_price", 0)

                if fair_value and tech_target:
                    deviation = abs(fair_value - tech_target) / current_price if current_price else 0
                    if deviation > 0.20:  # More than 20% difference
                        conflicts.append(
                            {
                                "type": "valuation_conflict",
                                "fundamental_target": fair_value,
                                "technical_target": tech_target,
                                "deviation": deviation,
                                "severity": "medium",
                            }
                        )

        # Check risk assessment conflicts
        if synthesis_input.sec_analysis and synthesis_input.fundamental_analysis:
            sec_risks = synthesis_input.sec_analysis.get("risks", [])
            fund_health = synthesis_input.fundamental_analysis.get("analysis", {}).get("health_score", 50)

            high_risk_count = sum(1 for risk in sec_risks if risk.get("severity") == "high")

            if high_risk_count > 3 and fund_health > 70:
                conflicts.append(
                    {
                        "type": "risk_assessment_conflict",
                        "sec_high_risks": high_risk_count,
                        "fundamental_health_score": fund_health,
                        "severity": "medium",
                    }
                )

        return conflicts

    async def _reconcile_conflicts(self, conflicts: List[Dict], synthesis_input: SynthesisInput) -> Dict:
        """Reconcile conflicts between analyses using reasoning"""
        symbol = synthesis_input.symbol

        if not conflicts:
            return {"status": "no_conflicts", "adjustments": {}}

        # Check if deterministic conflict resolution is enabled (saves tokens, faster)
        if self.use_deterministic and self.deterministic_conflict_resolution:
            self.logger.debug(f"{symbol} - Using deterministic conflict resolution (LLM bypass)")

            response_data = reconcile_conflicts(
                conflicts=conflicts,
                fundamental=synthesis_input.fundamental_analysis,
                technical=synthesis_input.technical_analysis,
                sec=synthesis_input.sec_analysis,
                market_context=synthesis_input.market_context,
                time_horizon="long_term",
            )

            return self._build_deterministic_response("conflict_resolution", response_data)

        # === LLM Path (fallback when deterministic is disabled) ===
        conflicts_json = json.dumps(conflicts, indent=2)
        schema_example = (
            "{\n"
            '  "reconciliation_strategy": [\n'
            "    {\n"
            '      "conflict_type": "recommendation_conflict",\n'
            '      "explanation": "...",\n'
            '      "prioritization": "...",\n'
            '      "weight_adjustments": {\n'
            '        "fundamental": 0.7,\n'
            '        "technical": 0.3\n'
            "      },\n"
            '      "rationale": "..."\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        prompt = (
            "Reconcile the following conflicts between different analyses:\n\n"
            f"Conflicts:\n{conflicts_json}\n\n"
            "For each conflict:\n"
            "1. Determine the most likely explanation\n"
            "2. Identify which analysis to prioritize and why\n"
            "3. Suggest weight adjustments for the final recommendation\n"
            "4. Provide reconciliation rationale\n\n"
            "Consider:\n"
            "- Time horizon differences (technical short-term vs fundamental long-term)\n"
            "- Market conditions and regime\n"
            "- Data quality and recency\n"
            "- Analysis confidence levels\n\n"
            "Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.\n\n"
            "Return a JSON object that strictly follows the schema below (values are illustrative):\n"
            f"{schema_example}\n"
        )

        response = await self.ollama.generate(
            model=self.models["reasoning"],
            prompt=prompt,
            system="Reconcile analytical conflicts with sound investment reasoning.",
            format="json",
        )

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["reasoning"],
            symbol=symbol,
            llm_type="synthesis_reconcile_conflicts",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Parse response safely
        parsed_response = self._parse_llm_response(response)

        return self._wrap_llm_response(
            response=parsed_response,
            model=self.models["reasoning"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

    async def _calculate_composite_scores(self, synthesis_input: SynthesisInput, analysis_scores: Dict) -> Dict:
        """Calculate composite scores from all analyses"""
        composite = {}

        # Base scores from each analysis
        scores = {"fundamental": 50, "technical": 50, "sec": 50, "sentiment": 50}

        # Extract scores from analyses
        if synthesis_input.fundamental_analysis:
            scores["fundamental"] = synthesis_input.fundamental_analysis.get("quality_score", 50)

        if synthesis_input.technical_analysis:
            tech_rating = synthesis_input.technical_analysis.get("technical_rating", 5)
            scores["technical"] = tech_rating * 10

        if synthesis_input.sec_analysis:
            sec_analysis = synthesis_input.sec_analysis.get("analysis", {})
            scores["sec"] = sec_analysis.get("overall_rating", 5) * 10

        if synthesis_input.sentiment_analysis:
            scores["sentiment"] = synthesis_input.sentiment_analysis.get("sentiment_score", 50)

        # Apply quality weights
        weighted_scores = {}
        for analysis_type, score in scores.items():
            quality = analysis_scores.get(analysis_type, 0.5)
            weighted_scores[analysis_type] = score * quality

        # Calculate weighted composite
        weights = {"fundamental": 0.35, "technical": 0.20, "sec": 0.30, "sentiment": 0.15}

        total_weight = sum(weights[k] for k in weighted_scores.keys())
        composite["overall_score"] = (
            sum(weighted_scores[k] * weights[k] for k in weighted_scores.keys()) / total_weight
            if total_weight > 0
            else 50
        )

        # Calculate confidence based on analysis completeness and agreement
        composite["confidence"] = self._calculate_confidence(synthesis_input, analysis_scores, weighted_scores)

        # Component scores
        composite["component_scores"] = weighted_scores
        composite["weights_applied"] = weights

        return composite

    async def _generate_investment_thesis(
        self, synthesis_input: SynthesisInput, key_insights: Dict, composite_scores: Dict
    ) -> Dict:
        """Generate comprehensive investment thesis"""
        # Check if deterministic thesis generation is enabled (saves tokens, faster)
        if self.use_deterministic and self.deterministic_thesis_generation:
            self.logger.debug(f"{synthesis_input.symbol} - Using deterministic thesis generation (LLM bypass)")

            response_data = generate_investment_thesis(
                symbol=synthesis_input.symbol,
                key_insights=key_insights,
                composite_scores=composite_scores,
                fundamental_analysis=synthesis_input.fundamental_analysis,
                company_profile=None,  # Will use data from fundamental_analysis if available
            )

            return self._build_deterministic_response("investment_thesis", response_data)

        # === LLM Path (fallback when deterministic is disabled) ===
        insights_json = json.dumps(key_insights, indent=2)[:4000]
        schema_example = (
            "{\n"
            '  "core_investment_narrative": "...",\n'
            '  "key_value_drivers": ["..."],\n'
            '  "competitive_advantages": ["..."],\n'
            '  "growth_catalysts": ["..."],\n'
            '  "bear_case_considerations": ["..."],\n'
            '  "time_horizon": "3-5 years",\n'
            '  "key_metrics_to_monitor": ["..."],\n'
            '  "thesis_invalidation_triggers": ["..."]\n'
            "}"
        )
        prompt = (
            f"Generate a comprehensive investment thesis for {synthesis_input.symbol}:\n\n"
            "Key Insights:\n"
            f"{insights_json}\n\n"
            "Composite Scores:\n"
            f"Overall Score: {composite_scores.get('overall_score', 50):.1f}/100\n"
            f"Confidence: {composite_scores.get('confidence', 50):.1f}%\n\n"
            "Create an investment thesis including:\n"
            "1. Core investment narrative (bull case)\n"
            "2. Key value drivers\n"
            "3. Competitive advantages\n"
            "4. Growth catalysts\n"
            "5. Bear case considerations\n"
            "6. Time horizon for the thesis to play out\n"
            "7. Key metrics to monitor\n"
            "8. Thesis invalidation triggers\n\n"
            "Be specific, actionable, and balanced.\n\n"
            "Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.\n\n"
            "Return a JSON object that strictly follows the schema below (values are illustrative):\n"
            f"{schema_example}\n"
        )

        response = await self.ollama.generate(
            model=self.models["synthesis"],
            prompt=prompt,
            system="Generate compelling yet balanced investment thesis.",
            format="json",
            prompt_name="_generate_investment_thesis_prompt",
        )

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["synthesis"],
            symbol=synthesis_input.symbol,
            llm_type="synthesis_investment_thesis",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Parse response safely
        parsed_response = self._parse_llm_response(response)

        return self._wrap_llm_response(
            response=parsed_response,
            model=self.models["synthesis"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

    async def _comprehensive_risk_assessment(self, synthesis_input: SynthesisInput, conflicts: List[Dict]) -> Dict:
        """Perform comprehensive risk assessment across all dimensions"""
        risks = {"categories": {}, "overall_risk": 50, "risk_reward_ratio": 1.0}

        # Aggregate risks from different analyses
        all_risks = []

        if synthesis_input.sec_analysis:
            sec_risks = synthesis_input.sec_analysis.get("risks", [])
            all_risks.extend(sec_risks)

        if synthesis_input.fundamental_analysis:
            fund_analysis = synthesis_input.fundamental_analysis.get("analysis", {})
            if "risks" in fund_analysis:
                all_risks.append(
                    {
                        "category": "financial",
                        "description": "Financial health concerns",
                        "severity": self._score_to_severity(100 - fund_analysis.get("health_score", 50)),
                    }
                )

        if synthesis_input.technical_analysis:
            tech_analysis = synthesis_input.technical_analysis.get("analysis", {})
            volatility = tech_analysis.get("volatility", 0)
            if volatility > 0.30:  # High volatility
                all_risks.append(
                    {
                        "category": "market",
                        "description": f"High volatility: {volatility:.1%}",
                        "severity": "high" if volatility > 0.50 else "medium",
                    }
                )

        # Add conflict-based risks
        if conflicts:
            all_risks.append(
                {
                    "category": "analysis",
                    "description": f"Conflicting signals across {len(conflicts)} analyses",
                    "severity": "medium",
                }
            )

        risks_json = json.dumps(all_risks, indent=2)[:4000]
        schema_example = (
            "{\n"
            '  "risk_categorization_and_prioritization": [\n'
            '    {"risk": "...", "category": "...", "priority": "High"}\n'
            "  ],\n"
            '  "risk_probability_and_impact_matrix": {\n'
            '    "market_competition": {"probability": "High", "impact": "High"}\n'
            "  },\n"
            '  "correlation_between_risks": "...",\n'
            '  "mitigation_strategies": "...",\n'
            '  "overall_risk_score": 75,\n'
            '  "risk_adjusted_return_potential": "...",\n'
            '  "worst_case_scenario_analysis": "...",\n'
            '  "black_swan_considerations": "..."\n'
            "}"
        )
        prompt = (
            "Perform a comprehensive risk assessment:\n\n"
            f"Identified Risks:\n{risks_json}\n\n"
            f"Symbol: {synthesis_input.symbol}\n\n"
            "Assess:\n"
            "1. Risk categorization and prioritization\n"
            "2. Risk probability and impact matrix\n"
            "3. Correlation between risks\n"
            "4. Mitigation strategies\n"
            "5. Overall risk score (0-100, higher = riskier)\n"
            "6. Risk-adjusted return potential\n"
            "7. Worst-case scenario analysis\n"
            "8. Black swan considerations\n\n"
            "Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.\n\n"
            "Return a JSON object that strictly follows the schema below (values are illustrative):\n"
            f"{schema_example}\n"
        )

        response = await self.ollama.generate(
            model=self.models["reasoning"],
            prompt=prompt,
            system="Perform thorough investment risk assessment.",
            format="json",
            prompt_name="_comprehensive_risk_assessment_prompt",
        )

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["reasoning"],
            symbol=synthesis_input.symbol,
            llm_type="synthesis_risk_assessment",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Parse response safely
        response_dict = self._parse_llm_response(response)
        risks.update(response_dict)

        return self._wrap_llm_response(
            response=risks, model=self.models["reasoning"], prompt=prompt, temperature=0.3, top_p=0.9, format="json"
        )

    async def _generate_scenarios(
        self, synthesis_input: SynthesisInput, composite_scores: Dict, risk_assessment: Dict
    ) -> Dict:
        """Generate bull, base, and bear case scenarios with smart valuation adjustments"""

        # FIX #5: Validate assessment values before scenario generation
        # Prevent all-zero inputs from reaching LLM (indicates data quality issues)
        overall_score = composite_scores.get("overall_score", 0)
        risk_score = risk_assessment.get("overall_risk", 0)

        if overall_score == 0 and risk_score == 0:
            self.logger.error(
                f"Cannot generate scenarios for {synthesis_input.symbol}: "
                f"All assessment values are zero (data quality issue). "
                f"Composite scores: {composite_scores}, Risk: {risk_assessment}"
            )
            return {
                "bull_case": {"error": "Insufficient data for scenario generation"},
                "base_case": {"error": "Insufficient data for scenario generation"},
                "bear_case": {"error": "Insufficient data for scenario generation"},
                "error": "All assessment values are zero - data quality issue",
            }

        # Apply smart valuation adjustments
        smart_targets = self._calculate_smart_price_targets(synthesis_input, composite_scores, risk_assessment)

        prompt = f"""
        Generate investment scenarios for {synthesis_input.symbol}:
        
        Current Assessment:
        - Overall Score: {composite_scores.get('overall_score', 50):.1f}/100
        - Risk Score: {risk_assessment.get('overall_risk', 50)}/100

        Smart Valuation Analysis:
        - Base Fair Value: ${synthesis_input.fundamental_analysis.get('fair_value', 0):.2f}
        - Multi-Model Blended: ${synthesis_input.fundamental_analysis.get('multi_model_summary', {}).get('blended_fair_value', 0):.2f}
        - Model Agreement Score: {synthesis_input.fundamental_analysis.get('multi_model_summary', {}).get('model_agreement_score', 0):.2f}
        - Technical Target: ${synthesis_input.technical_analysis.get('signals', {}).get('target_price', 0):.2f}
        - Smart Adjusted Target: ${smart_targets.get('adjusted_target', 0):.2f}
        - Current Price: ${synthesis_input.fundamental_analysis.get('valuation', {}).get('current_price', 0):.2f}
        - Valuation Bias: {smart_targets.get('valuation_summary', {}).get('valuation_bias', 'neutral')}
        - Quality Tier: {smart_targets.get('valuation_summary', {}).get('quality_tier', 'average')}
        
        Generate three scenarios:
        
        BULL CASE (30% probability):
        - Price target (12-month)
        - Key assumptions
        - Required catalysts
        - Upside potential
        
        BASE CASE (50% probability):
        - Price target (12-month)
        - Key assumptions
        - Expected developments
        - Return potential
        
        BEAR CASE (20% probability):
        - Price target (12-month)
        - Risk factors that materialize
        - Downside potential
        - Warning signs to watch
        
        Calculate probability-weighted expected return.

        Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.

        Return a JSON object containing keys:
        - "bull_case" (with probability, price_target, key_assumptions, required_catalysts, upside_potential)
        - "base_case" (with probability, price_target, key_assumptions, expected_developments, return_potential)
        - "bear_case" (with probability, price_target, risk_factors, downside_potential, warning_signs)
        - "probability_weighted_expected_return"
        """

        response = await self.ollama.generate(
            model=self.models["reasoning"],
            prompt=prompt,
            system="Generate realistic investment scenarios with clear assumptions.",
            format="json",
            prompt_name="_generate_scenarios_prompt",
        )

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["reasoning"],
            symbol=synthesis_input.symbol,
            llm_type="synthesis_scenarios",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Parse response safely with fallback
        try:
            parsed_response = self._parse_llm_response(response)

            # Validate scenarios have required fields
            if (
                not parsed_response.get("bull_case")
                or not parsed_response.get("base_case")
                or not parsed_response.get("bear_case")
            ):
                raise ValueError("Missing required scenario cases")

            return self._wrap_llm_response(
                response=parsed_response,
                model=self.models["reasoning"],
                prompt=prompt,
                temperature=0.3,
                top_p=0.9,
                format="json",
            )

        except Exception as e:
            # CRITICAL FIX #5: Smart fallback when scenario generation fails
            self.logger.warning(
                f"Scenario generation failed for {synthesis_input.symbol}: {e}. Using fallback scenarios."
            )

            current_price = synthesis_input.fundamental_analysis.get("valuation", {}).get("current_price", 100)
            fair_value = synthesis_input.fundamental_analysis.get("valuation", {}).get("fair_value", current_price)

            # Generate reasonable default scenarios based on fair value and current price
            upside_to_fair = ((fair_value - current_price) / current_price) if current_price > 0 else 0

            fallback_scenarios = {
                "bull_case": {
                    "price_target": round(current_price * 1.25, 2),
                    "probability": 30,
                    "upside_potential": 25.0,
                    "key_assumptions": [
                        "Market conditions improve",
                        "Company executes growth strategy",
                        "Valuation multiple expansion",
                    ],
                    "required_catalysts": [
                        "Strong earnings beat",
                        "Positive industry trends",
                        "Market sentiment improvement",
                    ],
                },
                "base_case": {
                    "price_target": round(fair_value, 2),
                    "probability": 50,
                    "return_potential": round(upside_to_fair * 100, 1),
                    "key_assumptions": [
                        "Current trends continue",
                        "Valuation normalizes to fair value",
                        "Stable market conditions",
                    ],
                    "expected_developments": ["Steady financial performance", "Modest growth in line with estimates"],
                },
                "bear_case": {
                    "price_target": round(current_price * 0.80, 2),
                    "probability": 20,
                    "downside_potential": -20.0,
                    "risk_factors": ["Market downturn", "Competitive pressure increases", "Economic headwinds"],
                    "warning_signs": ["Deteriorating fundamentals", "Market share loss", "Margin compression"],
                },
                "expected_return": round((0.3 * 25.0) + (0.5 * upside_to_fair * 100) + (0.2 * -20.0), 1),
                "methodology": "Fallback scenarios generated due to LLM parsing error",
                "error": str(e),
            }

            return self._wrap_llm_response(
                response=fallback_scenarios,
                model=self.models["reasoning"],
                prompt=prompt,
                temperature=0.3,
                top_p=0.9,
                format="json",
            )

    async def _make_recommendation(
        self, composite_scores: Dict, risk_assessment: Dict, scenarios: Dict, symbol: str
    ) -> Dict:
        """Make final investment recommendation"""
        # Unwrap scenarios if it's a wrapped LLM response
        if "response" in scenarios and isinstance(scenarios.get("response"), dict):
            scenarios_data = scenarios["response"]
        else:
            scenarios_data = scenarios

        overall_score = composite_scores.get("overall_score", 50)
        confidence = composite_scores.get("confidence", 50)
        risk_score = risk_assessment.get("overall_risk", 50)

        # Calculate risk-adjusted score
        risk_adjusted_score = overall_score * (1 - risk_score / 200)

        # Determine recommendation based on thresholds
        if risk_adjusted_score >= self.thresholds["strong_buy"]:
            recommendation = "strong_buy"
        elif risk_adjusted_score >= self.thresholds["buy"]:
            recommendation = "buy"
        elif risk_adjusted_score >= self.thresholds["hold"]:
            recommendation = "hold"
        elif risk_adjusted_score >= self.thresholds["sell"]:
            recommendation = "sell"
        else:
            recommendation = "strong_sell"

        # Get expected return from scenarios
        expected_return = scenarios_data.get("expected_return", 0)

        prompt = f"""
        Make final investment recommendation:
        
        Scores:
        - Overall Score: {overall_score:.1f}/100
        - Risk-Adjusted Score: {risk_adjusted_score:.1f}/100
        - Confidence: {confidence:.1f}%
        - Risk Level: {risk_score}/100
        
        Initial Recommendation: {recommendation}
        Expected Return: {expected_return:.1%}

        Scenarios:
        """

        # Append scenarios JSON outside f-string to avoid format specifier conflicts
        scenarios_json = json.dumps(scenarios_data, indent=2)[:2000]
        prompt += scenarios_json
        prompt += """

        Provide:
        1. Final recommendation (strong buy/buy/hold/sell/strong sell)
        2. Conviction level (high/medium/low)
        3. Position sizing suggestion (% of portfolio)
        4. Time horizon (short/medium/long term)
        5. Key reasons for recommendation (top 3)
        6. Main risks to monitor (top 3)
        7. Exit conditions
        
        Be decisive but acknowledge uncertainty.

        Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.

        Return as structured JSON, wrapped in a markdown code block (```json ... ```). For example:
        ```json
        {{
          "final_recommendation": "buy",
          "conviction_level": "high",
          "position_sizing_suggestion": 5,
          "time_horizon": "long term",
          "key_reasons_for_recommendation": [
            "Strong fundamentals",
            "Attractive valuation",
            "Positive technical momentum"
          ],
          "main_risks_to_monitor": [
            "Increased competition",
            "Regulatory changes",
            "Execution risk"
          ],
          "exit_conditions": [
            "A sustained break below the 200-day moving average",
            "A significant deterioration in fundamentals"
          ]
        }}
        ```
        """

        response = await self.ollama.generate(
            model=self.models["decision"],
            prompt=prompt,
            format="json",
            prompt_name="_make_recommendation_prompt",
        )

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["decision"],
            symbol=symbol,
            llm_type="synthesis_recommendation",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Parse response safely
        response_dict = self._parse_llm_response(response)
        response_dict["risk_adjusted_score"] = risk_adjusted_score
        response_dict["expected_return"] = expected_return

        return self._wrap_llm_response(
            response=response_dict,
            model=self.models["decision"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

    async def _generate_action_plan(
        self, recommendation: Dict, synthesis_input: SynthesisInput, scenarios: Dict
    ) -> Dict:
        """Generate specific action plan based on recommendation"""
        prompt = f"""
        Generate specific action plan for {synthesis_input.symbol}:
        
        Recommendation: {recommendation.get('final_recommendation')}
        Conviction: {recommendation.get('conviction_level')}
        Position Size: {recommendation.get('position_sizing')}%
        
        Technical Levels:
        {json.dumps(synthesis_input.technical_analysis.get('levels', {}), indent=2) if synthesis_input.technical_analysis else 'N/A'}
        
        Create actionable plan:
        1. Entry strategy
           - Ideal entry price range
           - Entry timing considerations
           - Position building approach (all at once vs scaling)
        
        2. Risk management
           - Stop loss level and rationale
           - Position size adjustments
           - Hedging considerations
        
        3. Profit taking
           - Target levels (multiple)
           - Scaling out strategy
           - Rebalancing triggers
        
        4. Monitoring plan
           - Key metrics to track
           - Review frequency
           - Alert conditions
        
        5. Contingency plans
           - If price drops 10%
           - If thesis changes
           - If better opportunity arises
        
        Be specific with prices, percentages, and conditions.

        Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.

        Return as structured JSON, wrapped in a markdown code block (```json ... ```). For example:
        """

        # Append JSON example outside f-string to avoid format specifier conflicts
        json_example = """
        ```json
        {
          "entry_strategy": {
            "ideal_entry_price_range": [140.0, 145.0],
            "entry_timing_considerations": "Wait for a pullback to the 50-day moving average.",
            "position_building_approach": "Scale in with 3 separate purchases."
          },
          "risk_management": {
            "stop_loss_level": 135.0,
            "rationale": "A break below this level would invalidate the bullish thesis.",
            "position_size_adjustments": "Reduce position size if the stock price drops below $140.",
            "hedging_considerations": "Consider buying put options to hedge against a market downturn."
          },
          "profit_taking": {
            "target_levels": [160.0, 170.0],
            "scaling_out_strategy": "Sell 50% of the position at the first target and the remaining 50% at the second target.",
            "rebalancing_triggers": "Rebalance the position if it exceeds 10% of the portfolio."
          },
          "monitoring_plan": {
            "key_metrics_to_track": ["Revenue growth", "Net margin", "Free cash flow"],
            "review_frequency": "Quarterly",
            "alert_conditions": "A sustained decline in revenue growth for two consecutive quarters."
          },
          "contingency_plans": {
            "if_price_drops_10_percent": "Review the position and consider reducing the position size.",
            "if_thesis_changes": "Exit the position.",
            "if_better_opportunity_arises": "Consider selling a portion of the position to fund the new opportunity."
          }
        }
        ```
        """
        prompt += json_example

        response = await self.ollama.generate(
            model=self.models["synthesis"],
            prompt=prompt,
            system="Generate specific, actionable investment plan.",
            format="json",
            prompt_name="_generate_action_plan_prompt",
        )

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["synthesis"],
            symbol=synthesis_input.symbol,
            llm_type="synthesis_action_plan",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Parse response safely
        parsed_response = self._parse_llm_response(response)

        return self._wrap_llm_response(
            response=parsed_response,
            model=self.models["synthesis"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

    async def _create_synthesis_report(self, report_data: Dict) -> Dict:
        """Create final comprehensive synthesis report"""
        # Round numeric values to reduce token usage
        from investigator.domain.services.data_normalizer import DataNormalizer

        rounded_data = DataNormalizer.round_financial_data(report_data)

        # Check if TOON format is enabled
        use_toon = getattr(self.config.ollama, "use_toon_format", False) and getattr(
            self.config.ollama, "toon_agents", {}
        ).get("synthesis", False)

        # Format data section (TOON or JSON)
        if use_toon:
            # Extract peer comparison data for TOON formatting (55% token savings)
            peer_data = rounded_data.get("peer_comparison", {}).get("peers", [])

            if peer_data and isinstance(peer_data, list) and len(peer_data) > 0:
                try:
                    # Convert peer comparison to TOON format
                    toon_peers = to_toon_peers(peer_data)

                    # Remove peer_comparison from rounded_data to avoid duplication
                    remaining_data = rounded_data.copy()
                    if "peer_comparison" in remaining_data:
                        peer_comp_copy = remaining_data["peer_comparison"].copy()
                        peer_comp_copy.pop("peers", None)
                        remaining_data["peer_comparison"] = peer_comp_copy

                    # Build data section with TOON peers + JSON for other data
                    data_section = (
                        f"{toon_peers}\n\nAdditional Analysis:\n{json.dumps(remaining_data, indent=2)[:8000]}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to convert peer data to TOON: {e}")
                    data_section = json.dumps(rounded_data, indent=2)[:10000]
            else:
                # No peer data, use JSON
                data_section = json.dumps(rounded_data, indent=2)[:10000]
        else:
            # TOON disabled, use JSON (current behavior)
            data_section = json.dumps(rounded_data, indent=2)[:10000]

        prompt = f"""
        Create executive investment report for {rounded_data['symbol']}:

        {data_section}
        
        Structure the report as:
        
        1. EXECUTIVE SUMMARY (3-5 bullet points)
           - Investment recommendation
           - Key thesis points
           - Risk-return profile
           - Time horizon
        
        2. INVESTMENT THESIS
           - Bull case narrative
           - Value drivers
           - Catalysts
        
        3. VALUATION SUMMARY
           - Current vs fair value
           - Multiple scenarios
           - Margin of safety
        
        4. RISK ANALYSIS
           - Primary risks
           - Mitigation factors
           - Monitoring triggers
        
        5. TECHNICAL PERSPECTIVE
           - Trend analysis (including daily and weekly perspectives)
           - Key levels (support/resistance)
           - Pattern interpretation (daily and weekly)
           - Timing considerations
        
        6. RECOMMENDATION & ACTION PLAN
           - Clear recommendation
           - Specific actions
           - Position management
        
        7. APPENDIX
           - Key metrics summary
           - Peer comparison highlights
           - Data quality notes
        
        Make it concise, professional, and actionable.

        Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.

        Return as structured JSON, wrapped in a markdown code block (```json ... ```). For example:
        ```json
        {{
          "executive_summary": [
            "Recommendation: Buy",
            "Thesis: Strong growth and undervaluation",
            "Risk/Reward: Favorable",
            "Time Horizon: 12-18 months"
          ],
          "investment_thesis": {{
            "bull_case_narrative": "The company is a market leader in a growing industry, with a strong brand and a loyal customer base. The stock is currently undervalued and offers an attractive risk/reward profile.",
            "value_drivers": ["Strong revenue growth", "Expanding margins"],
            "catalysts": ["New product launches", "Expansion into new geographic markets"]
          }},
          "valuation_summary": {{
            "current_vs_fair_value": "The stock is currently trading at a 20% discount to our fair value estimate of $150.",
            "scenarios": "Our bull case scenario suggests a price target of $200, while our bear case scenario suggests a price target of $100.",
            "margin_of_safety": "The current stock price offers a 20% margin of safety."
          }},
          "risk_analysis": {{
            "primary_risks": ["Increased competition", "Regulatory changes"],
            "mitigation_factors": "The company is diversifying its product portfolio to mitigate the risk of competition.",
            "monitoring_triggers": "A sustained decline in revenue growth for two consecutive quarters."
          }},
          "technical_perspective": {{
            "trend_analysis": "The stock is in a strong uptrend.",
            "key_levels": {{ "support": 140.0, "resistance": 155.0 }},
            "timing_considerations": "Wait for a pullback to the 50-day moving average before entering a position."
          }},
          "recommendation_and_action_plan": {{
            "recommendation": "Buy",
            "actions": ["Initiate a position at current levels", "Set a stop-loss at $135"],
            "position_management": "Scale into the position with 3 separate purchases."
          }},
          "appendix": {{
            "key_metrics_summary": {{ "P/E": 25, "P/S": 5, "ROE": 0.20 }},
            "peer_comparison_highlights": "The company is outperforming its peers in terms of revenue growth and profitability.",
            "data_quality_notes": "The financial data used in this analysis is of high quality."
          }}
        }}
        ```
        """

        # Build system prompt with optional TOON explanation
        system_prompt = "Create professional investment report with clear recommendations."
        if use_toon and peer_data:
            system_prompt += "\n\n" + TOONFormatter.get_format_explanation()

        response = await self.ollama.generate(
            model=self.models["synthesis"],
            prompt=prompt,
            system=system_prompt,
            format="json",
            prompt_name="_create_synthesis_report_prompt",
        )

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["synthesis"],
            symbol=report_data["symbol"],
            llm_type="synthesis_report_1",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Parse response safely
        response_dict = self._parse_llm_response(response)
        response_dict["generated_at"] = datetime.now().isoformat()
        response_dict["synthesis_version"] = "2.0"

        return self._wrap_llm_response(
            response=response_dict,
            model=self.models["synthesis"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

    def _evaluate_analysis_quality(self, analysis: Dict, required_fields: List[str]) -> float:
        """Evaluate the quality and completeness of an analysis"""
        if not analysis:
            return 0.0

        # Check for required fields
        completeness = sum(1 for field in required_fields if field in analysis) / len(required_fields)

        # Check for errors
        if analysis.get("status") == "error":
            return 0.0

        # Check data freshness (if timestamp available)
        freshness = 1.0  # Default to fresh

        # Combine scores
        quality_score = completeness * 0.7 + freshness * 0.3

        return quality_score

    def _prepare_analysis_summary(self, synthesis_input: SynthesisInput) -> Dict:
        """Prepare summary of all analyses for LLM processing"""
        summary = {}

        if synthesis_input.sec_analysis:
            summary["sec"] = {
                "risks": synthesis_input.sec_analysis.get("risks", [])[:5],
                "metrics": synthesis_input.sec_analysis.get("metrics", {}),
                "key_points": synthesis_input.sec_analysis.get("analysis", {}).get("executive_summary", ""),
            }

        if synthesis_input.fundamental_analysis:
            summary["fundamental"] = {
                "valuation": synthesis_input.fundamental_analysis.get("valuation", {}),
                "quality_score": synthesis_input.fundamental_analysis.get("quality_score", 0),
                "key_points": synthesis_input.fundamental_analysis.get("analysis", {}).get("investment_thesis", ""),
            }

        if synthesis_input.technical_analysis:
            summary["technical"] = {
                "signals": synthesis_input.technical_analysis.get("signals", {}),
                "levels": synthesis_input.technical_analysis.get("levels", {}),
                "recommendation": synthesis_input.technical_analysis.get("recommendation", ""),
            }

        if synthesis_input.sentiment_analysis:
            summary["sentiment"] = synthesis_input.sentiment_analysis

        return summary

    def _extract_quantitative_insights(self, synthesis_input: SynthesisInput) -> Dict:
        """Extract quantitative insights from analyses"""
        insights = {}

        # Extract valuation metrics
        if synthesis_input.fundamental_analysis:
            valuation = synthesis_input.fundamental_analysis.get("valuation", {})
            current_price = valuation.get("current_price", 0)
            fair_value = valuation.get("fair_value", 0)

            if current_price and fair_value:
                insights["upside_potential"] = ((fair_value - current_price) / current_price) * 100
                insights["margin_of_safety"] = ((current_price - fair_value) / fair_value) * 100

        # Extract technical metrics
        if synthesis_input.technical_analysis:
            signals = synthesis_input.technical_analysis.get("signals", {})
            insights["technical_confidence"] = signals.get("confidence_level", 0)
            insights["risk_reward_ratio"] = signals.get("risk_reward_ratio", 1)

        return insights

    def _are_recommendations_conflicting(self, rec1: str, rec2: str) -> bool:
        """Check if two recommendations are conflicting"""
        bullish = ["strong_buy", "buy"]
        bearish = ["sell", "strong_sell"]

        return (rec1 in bullish and rec2 in bearish) or (rec1 in bearish and rec2 in bullish)

    def _score_to_severity(self, score: float) -> str:
        """Convert numerical score to severity level"""
        if score >= 70:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"

    def _calculate_confidence(
        self, synthesis_input: SynthesisInput, analysis_scores: Dict, weighted_scores: Dict
    ) -> float:
        """Calculate overall confidence in the analysis"""
        confidence_factors = []

        # Factor 1: Analysis completeness
        available_analyses = sum(
            1
            for attr in ["sec_analysis", "fundamental_analysis", "technical_analysis", "sentiment_analysis"]
            if getattr(synthesis_input, attr) is not None
        )
        completeness = available_analyses / 4
        confidence_factors.append(completeness)

        # Factor 2: Analysis quality
        avg_quality = np.mean(list(analysis_scores.values())) if analysis_scores else 0.5
        confidence_factors.append(avg_quality)

        # Factor 3: Agreement between analyses
        if len(weighted_scores) > 1:
            score_std = np.std(list(weighted_scores.values()))
            agreement = 1 - min(score_std / 50, 1)  # Normalize standard deviation
            confidence_factors.append(agreement)

        # Calculate weighted confidence
        confidence = np.mean(confidence_factors) * 100

        return min(max(confidence, 0), 100)  # Clamp between 0 and 100

    async def generate_peer_synthesis(self, target: str, peers: List[str], analyses: Dict[str, Dict]) -> Dict:
        """Generate synthesis comparing target to peers"""
        # Round numeric values to reduce token usage
        from investigator.domain.services.data_normalizer import DataNormalizer

        rounded_analyses = DataNormalizer.round_financial_data(analyses)

        prompt = f"""
        Synthesize peer comparison analysis:

        Target Company: {target}
        Peer Companies: {', '.join(peers)}

        Analyses:
        {json.dumps(rounded_analyses, indent=2)[:8000]}
        
        Provide:
        1. Relative ranking (1 to {len(peers) + 1})
        2. Competitive advantages/disadvantages
        3. Relative valuation assessment
        4. Best-in-class metrics
        5. Investment preference ranking
        6. Pair trade opportunities
        7. Sector outlook implications
        
        Return structured JSON with clear comparisons.
        """

        response = await self.ollama.generate(
            model=self.models["synthesis"],
            prompt=prompt,
            system="Synthesize peer comparison for investment decisions.",
            format="json",
            prompt_name="generate_peer_synthesis_prompt",
        )

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["synthesis"],
            symbol=target,
            llm_type="synthesis_peer_comparison",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Parse response safely
        parsed_response = self._parse_llm_response(response)

        return self._wrap_llm_response(
            response=parsed_response,
            model=self.models["synthesis"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )
