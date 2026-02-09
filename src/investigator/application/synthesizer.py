#!/usr/bin/env python3
"""
InvestiGator - Analysis Synthesis Module (Refactored)
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

This module synthesizes fundamental and technical analysis to generate final investment
recommendations. Report generation and charting are delegated to separate modules.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from sqlalchemy import text

from investigator.config import get_config

# Import from Clean Architecture
from investigator.domain.models import InvestmentRecommendation
from investigator.infrastructure.cache import CacheManager, CacheType, get_cache_manager
from investigator.infrastructure.database.db import (
    DatabaseManager,
    get_llm_responses_dao,
)
from investigator.infrastructure.reporting import (
    PDFReportGenerator,
    ReportConfig,
    WeeklyReportGenerator,
)
from investigator.infrastructure.ui import ASCIIArt
from investigator.domain.services.analysis.peer_comparison import get_peer_comparison_analyzer
from investigator.infrastructure.llm.llm_facade import create_llm_facade

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class _StubChartGenerator:
    """Fallback chart generator that disables chart creation to avoid crashes."""

    def __init__(self, charts_dir):
        self.charts_dir = Path(charts_dir)
        self.logger = logging.getLogger(f"{__name__}.ChartGeneratorStub")
        self.logger.warning("Chart generation disabled – matplotlib backend unavailable or failed to initialize.")

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            self.logger.debug("Chart generator stub invoked for method '%s'", name)
            return ""

        return _noop


# Attempt to load the real chart generator only when explicitly enabled.
if os.environ.get("INVESTIGATOR_ENABLE_CHARTS", "0") == "1":
    try:
        from utils.chart_generator import ChartGenerator as _RealChartGenerator

        ChartGenerator = _RealChartGenerator
    except Exception as chart_import_error:
        logger.warning(
            "Failed to import chart generator (%s). Falling back to stub implementation.",
            chart_import_error,
        )
        ChartGenerator = _StubChartGenerator
else:
    ChartGenerator = _StubChartGenerator


# InvestmentRecommendation is now imported from investigator.domain.models


class InvestmentSynthesizer:
    """Synthesizes fundamental and technical analysis into actionable recommendations"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize synthesizer with configuration (migrated from config.json)"""
        self.config = get_config()
        self.db_manager = DatabaseManager()

        # Initialize interfaces
        self.ollama = create_llm_facade(self.config, cache_manager=None)
        # Import CacheManager here to avoid circular imports
        from investigator.infrastructure.cache import CacheManager

        self.cache_manager = CacheManager(self.config)  # Use cache manager with config

        # Initialize generators
        self.chart_generator = ChartGenerator(self.config.reports_dir / "charts")
        self.report_generator = PDFReportGenerator(
            self.config.reports_dir / "synthesis",
            ReportConfig(
                title="Investment Analysis Report", subtitle="Comprehensive Stock Analysis", include_charts=True
            ),
        )
        self.weekly_report_generator = WeeklyReportGenerator(self.config.reports_dir / "weekly")

        # Initialize DAOs
        self.llm_dao = get_llm_responses_dao()

        # Initialize loggers first (used by alert system)
        self.main_logger = self.config.get_main_logger("synthesizer")

        # Initialize alert system (Tier 3 Enhancement #14)
        from utils.alert_engine import AlertEngine
        from utils.email_notifier import EmailNotifier

        self.alert_engine = AlertEngine(db_manager=self.db_manager)

        # Initialize email notifier if configured
        self.email_notifier = None
        if hasattr(self.config, "email") and self.config.email:
            email_config = self.config.email
            # Check if EmailConfig has required attributes and is enabled
            if (
                hasattr(email_config, "enabled")
                and email_config.enabled
                and hasattr(email_config, "smtp_server")
                and hasattr(email_config, "smtp_port")
                and hasattr(email_config, "from_address")
                and hasattr(email_config, "password")
            ):
                try:
                    self.email_notifier = EmailNotifier(
                        smtp_host=email_config.smtp_server,
                        smtp_port=email_config.smtp_port,
                        sender_email=email_config.from_address,
                        sender_password=email_config.password,
                    )
                    self.main_logger.info("Email notifier initialized for alerts")
                except Exception as e:
                    self.main_logger.warning(f"Failed to initialize email notifier: {e}")

        # Response processing handled by LLM facade

        # Cache directories
        self.llm_cache_dir = self.config.data_dir / "llm_cache"
        self.llm_cache_dir.mkdir(parents=True, exist_ok=True)

        self.main_logger.info("Investment synthesizer initialized")

    def _get_latest_fiscal_period(self, fundamental_data=None, technical_data=None):
        """
        Determine the latest fiscal period from available data.

        Args:
            fundamental_data: SEC fundamental data (optional)
            technical_data: Technical analysis data (optional)

        Returns:
            Tuple of (fiscal_year, fiscal_period)
        """
        try:
            # For now, use current year and determine quarter based on current date
            from datetime import datetime

            current_date = datetime.now()
            current_year = current_date.year

            # Determine fiscal quarter based on current month
            month = current_date.month
            if month <= 3:
                fiscal_period = "Q4"  # Q4 of previous year
                fiscal_year = current_year - 1
            elif month <= 6:
                fiscal_period = "Q1"
                fiscal_year = current_year
            elif month <= 9:
                fiscal_period = "Q2"
                fiscal_year = current_year
            else:
                fiscal_period = "Q3"
                fiscal_year = current_year

            # TODO: In the future, extract this from actual SEC filing data
            # if fundamental_data and 'fiscal_year' in fundamental_data:
            #     fiscal_year = fundamental_data['fiscal_year']
            #     fiscal_period = fundamental_data.get('fiscal_period', fiscal_period)

            return fiscal_year, fiscal_period

        except Exception as e:
            self.main_logger.warning(f"Could not determine fiscal period: {e}, using defaults")
            return datetime.now().year, "FY"

    def _infer_period_from_fundamental(self, llm_responses: Dict[str, Dict]) -> Optional[str]:
        """Extract fiscal period label directly from cached fundamental analyses if available."""
        fundamentals = llm_responses.get("fundamental") or {}
        if not fundamentals:
            return None

        def _candidate_dicts() -> List[Dict[str, Any]]:
            for key, payload in fundamentals.items():
                if isinstance(payload, dict):
                    yield payload
                    meta = payload.get("metadata")
                    if isinstance(meta, dict):
                        yield meta
                    content = payload.get("content")
                    if isinstance(content, dict):
                        yield content

        for candidate in _candidate_dicts():
            fiscal_year = candidate.get("fiscal_year")
            fiscal_period = candidate.get("fiscal_period") or candidate.get("period")
            period_label = candidate.get("fiscal_period_label") or candidate.get("period_label")
            if fiscal_year and fiscal_period:
                return f"{fiscal_year}-{fiscal_period}"
            if period_label:
                return str(period_label)
        return None

    def synthesize_analysis(self, symbol: str, synthesis_mode: str = "comprehensive") -> InvestmentRecommendation:
        """
        Synthesize fundamental and technical analysis for a symbol

        Args:
            symbol: Stock symbol to analyze

        Returns:
            InvestmentRecommendation object
        """
        try:
            start_time = time.time()

            # Get symbol-specific logger
            symbol_logger = self.config.get_symbol_logger(symbol, "synthesizer")

            self.main_logger.info(f"Starting synthesis for {symbol}")
            symbol_logger.info("Starting investment synthesis")

            # Fetch all LLM responses for the symbol
            symbol_logger.info("Fetching LLM analysis responses")
            llm_responses = self._fetch_llm_responses(symbol)
            symbol_logger.debug(
                f"LLM responses keys: {list(llm_responses.keys()) if isinstance(llm_responses, dict) else type(llm_responses)}"
            )

            # Fetch latest market data
            symbol_logger.info("Fetching latest market data")
            latest_data = self._fetch_latest_data(symbol)

            # Fetch historical scores for trend analysis
            symbol_logger.info("Fetching historical investment scores")
            score_history = self._fetch_historical_scores(symbol)
            score_trend = self._calculate_score_trend(score_history) if score_history else {}

            if score_trend and score_trend.get("trend") != "insufficient_data":
                symbol_logger.info(
                    f"Score trend: {score_trend.get('trend')} ({score_trend.get('change'):+.1f} points, "
                    f"{score_trend.get('change_percent'):+.1f}%)"
                )

            # Fetch quarterly metrics for trend analysis (12 quarters for geometric mean)
            symbol_logger.info("Fetching quarterly metrics for trend analysis")
            quarterly_metrics = self._fetch_quarterly_metrics(symbol, limit=12)
            quarterly_trends = self._calculate_quarterly_trends(quarterly_metrics) if quarterly_metrics else {}

            # Log trend summary
            if quarterly_trends:
                qoq = quarterly_trends.get("qoq_growth", {})
                yoy = quarterly_trends.get("yoy_growth", {})
                symbol_logger.info(
                    f"Quarterly trends calculated: Revenue Q-o-Q: {qoq.get('revenue', 'N/A')}%, "
                    f"Net Income Q-o-Q: {qoq.get('net_income', 'N/A')}%"
                )

            # Fetch multi-year historical trends
            symbol_logger.info("Fetching multi-year historical financials (5 years)")
            multi_year_data = self._fetch_multi_year_financials(symbol, years=5)
            if multi_year_data and len(multi_year_data) >= 2:
                multi_year_metrics = self._calculate_multi_year_metrics(multi_year_data)
                symbol_logger.info(
                    f"Multi-year: Revenue CAGR {multi_year_metrics.get('revenue_cagr', 'N/A')}%, "
                    f"Pattern: {multi_year_metrics.get('cyclical_pattern', 'N/A')}, "
                    f"{len(multi_year_data)} years analyzed"
                )
                multi_year_trends = {"data": multi_year_data, "metrics": multi_year_metrics}
            else:
                multi_year_trends = None
                symbol_logger.warning(
                    f"Insufficient multi-year data for {symbol} (need at least 2 years, have {len(multi_year_data) if multi_year_data else 0})"
                )

            # Tier 3: DCF Valuation Analysis
            inferred_period = self._infer_period_from_fundamental(llm_responses)
            if inferred_period:
                current_period_label = inferred_period
            else:
                fy, fp = self._get_latest_fiscal_period()
                current_period_label = f"{fy}-{fp}" if fy and fp else None
            dcf_cache_key = {"symbol": symbol, "llm_type": "deterministic_dcf"}
            if current_period_label:
                dcf_cache_key["period"] = current_period_label

            dcf_valuation = None
            cached_dcf = self.cache_manager.get(CacheType.LLM_RESPONSE, dcf_cache_key)
            if cached_dcf:
                dcf_valuation = cached_dcf.get("response") or cached_dcf
                symbol_logger.info("Reusing cached deterministic DCF valuation from fundamental agent.")

            if quarterly_metrics and len(quarterly_metrics) >= 4 and dcf_valuation is None:
                try:
                    symbol_logger.info("Calculating DCF (Discounted Cash Flow) valuation with unified terminal growth")
                    from investigator.domain.services.fcf_growth_calculator import FCFGrowthCalculator
                    from investigator.domain.services.terminal_growth_calculator import TerminalGrowthCalculator
                    from investigator.domain.services.valuation.dcf import DCFValuation
                    from investigator.domain.services.valuation_framework_planner import ValuationFrameworkPlanner

                    # Step 1: Create DCF instance
                    dcf_analyzer = DCFValuation(
                        symbol=symbol,
                        quarterly_metrics=quarterly_metrics,
                        multi_year_data=multi_year_data if multi_year_data else [],
                        db_manager=self.db_manager,
                    )

                    # Step 2: Calculate Rule of 40 to get metrics for terminal growth
                    rule_of_40_result = dcf_analyzer._calculate_rule_of_40()
                    rule_of_40_score = rule_of_40_result.get("score", 0)
                    revenue_growth_pct = rule_of_40_result.get("revenue_growth_pct", 0)
                    profit_margin_pct = rule_of_40_result.get("profit_margin_pct", 0)

                    # Step 3: Calculate FCF margin
                    fcf_calc = FCFGrowthCalculator(symbol)
                    fcf_margin_pct = fcf_calc.calculate_fcf_margin(quarterly_metrics, ttm=True)

                    # Step 4: Get sector and market cap for ValuationFrameworkPlanner
                    sector = dcf_analyzer.sector
                    # Get market cap from latest quarterly metrics
                    market_cap_billions = 0.0
                    if quarterly_metrics:
                        latest_market_cap = quarterly_metrics[-1].get("market_cap", 0)
                        market_cap_billions = latest_market_cap / 1e9 if latest_market_cap > 0 else 0.0

                    # Step 5: Create ValuationFrameworkPlanner
                    planner = ValuationFrameworkPlanner(
                        symbol=symbol,
                        sector=sector,
                        industry="",  # Not critical for classification
                        market_cap_billions=market_cap_billions,
                    )

                    # Step 6: Classify company stage
                    company_stage = planner.classify_company_stage(
                        revenue_growth_pct=revenue_growth_pct, fcf_margin_pct=fcf_margin_pct
                    )

                    # Step 7: Create TerminalGrowthCalculator
                    terminal_calc = TerminalGrowthCalculator(
                        symbol=symbol, sector=sector, base_terminal_growth=0.035  # 3.5% base for tech
                    )

                    # Step 8: Calculate unified terminal growth
                    terminal_result = terminal_calc.calculate_terminal_growth(
                        rule_of_40_score=rule_of_40_score,
                        revenue_growth_pct=revenue_growth_pct,
                        fcf_margin_pct=fcf_margin_pct,
                    )
                    terminal_growth_rate = terminal_result["terminal_growth_rate"]

                    symbol_logger.info(
                        f"Unified Terminal Growth: {terminal_growth_rate*100:.2f}% "
                        f"(base: {terminal_result['base_rate']*100:.2f}% + "
                        f"quality: {terminal_result['adjustment']*100:+.2f}%) | "
                        f"Tier: {terminal_result['tier']} | {terminal_result['reason']}"
                    )

                    # Step 9: Calculate DCF with unified terminal growth rate
                    dcf_valuation = dcf_analyzer.calculate_dcf_valuation(terminal_growth_rate=terminal_growth_rate)

                    if dcf_valuation:
                        fair_value = dcf_valuation.get("fair_value_per_share", 0)
                        upside = dcf_valuation.get("upside_downside_pct", 0)
                        assessment = dcf_valuation.get("valuation_assessment", "Unknown")
                        symbol_logger.info(
                            f"DCF Fair Value: ${fair_value:.2f}, Upside: {upside:+.1f}%, Assessment: {assessment}"
                        )
                        cache_payload = {
                            "response": dcf_valuation,
                            "metadata": {
                                "cached_at": datetime.now(timezone.utc).isoformat(),
                                "analysis_type": "deterministic_dcf",
                                "source": "synthesizer",
                                "period": current_period_label,
                            },
                        }
                        self.cache_manager.set(CacheType.LLM_RESPONSE, dcf_cache_key, cache_payload)
                except Exception as e:
                    symbol_logger.error(f"Error calculating DCF valuation: {e}")
                    dcf_valuation = None
            elif dcf_valuation is None:
                symbol_logger.info("Skipping DCF valuation - insufficient quarterly data")

            # Tier 3: Recession Performance Analysis
            recession_performance = None
            if multi_year_data and len(multi_year_data) >= 5:
                try:
                    symbol_logger.info("Analyzing recession performance")
                    recession_performance = self._analyze_recession_performance(symbol, multi_year_data)

                    if recession_performance:
                        defensive_score = recession_performance.get("defensive_score", 5.0)
                        defensive_rating = recession_performance.get("defensive_rating", "Unknown")
                        symbol_logger.info(f"Defensive characteristics: {defensive_rating} ({defensive_score}/10)")
                except Exception as e:
                    symbol_logger.error(f"Error analyzing recession performance: {e}")
                    recession_performance = None
            else:
                symbol_logger.info("Skipping recession performance - need at least 5 years of historical data")

            # Tier 3: Insider Trading Analysis
            insider_trading = None
            try:
                symbol_logger.info("Analyzing insider trading activity")
                from utils.insider_trading import InsiderTradingAnalyzer

                insider_analyzer = InsiderTradingAnalyzer(db_manager=self.db_manager)
                insider_trading = insider_analyzer.analyze_insider_activity(symbol, days=180)

                if insider_trading and insider_trading.get("data_available"):
                    sentiment_score = insider_trading.get("sentiment_score", 5.0)
                    sentiment_rating = insider_trading.get("sentiment_rating", "Neutral")
                    buy_count = insider_trading.get("buy_count", 0)
                    sell_count = insider_trading.get("sell_count", 0)
                    symbol_logger.info(
                        f"Insider sentiment: {sentiment_rating} ({sentiment_score}/10) - {buy_count} buys, {sell_count} sells"
                    )
                else:
                    symbol_logger.info("Insider trading data not yet available - requires SEC Form 4 integration")
            except Exception as e:
                symbol_logger.error(f"Error analyzing insider trading: {e}")
                insider_trading = None

            # Tier 3: News Sentiment Analysis
            news_sentiment = None
            try:
                symbol_logger.info("Analyzing news sentiment")
                from utils.news_sentiment import NewsSentimentAnalyzer

                # Initialize with Ollama client for LLM-powered sentiment
                news_analyzer = NewsSentimentAnalyzer(db_manager=self.db_manager, ollama_client=self.ollama)
                news_sentiment = news_analyzer.analyze_news_sentiment(symbol, days=7)

                if news_sentiment and news_sentiment.get("data_available"):
                    sentiment_score = news_sentiment.get("sentiment_score", 5.0)
                    sentiment_rating = news_sentiment.get("sentiment_rating", "Neutral")
                    article_count = news_sentiment.get("article_count", 0)
                    trend = news_sentiment.get("sentiment_trend", {})
                    trend_dir = trend.get("direction", "stable")
                    symbol_logger.info(
                        f"News sentiment: {sentiment_rating} ({sentiment_score}/10) from {article_count} articles, trend: {trend_dir}"
                    )
                else:
                    symbol_logger.info("News sentiment data not yet available - requires NewsAPI integration")
            except Exception as e:
                symbol_logger.error(f"Error analyzing news sentiment: {e}")
                news_sentiment = None

            # Fetch peer valuation metrics
            symbol_logger.info("Fetching peer valuation metrics")
            peer_valuation = self._fetch_peer_valuation_metrics(symbol)

            if peer_valuation and peer_valuation.get("relative_valuation"):
                rel_val = peer_valuation["relative_valuation"]
                assessment = rel_val.get("overall_assessment", "unknown")
                symbol_logger.info(f"Relative valuation: {assessment}")

            # Detect red flags
            symbol_logger.info("Detecting red flags in financial data")
            red_flags = self._detect_red_flags(symbol, quarterly_metrics, quarterly_trends, latest_data)
            if red_flags:
                high_severity = [f for f in red_flags if f["severity"] == "high"]
                symbol_logger.warning(f"⚠️  Detected {len(red_flags)} red flags ({len(high_severity)} high severity)")

            # Extract support/resistance levels
            support_resistance = latest_data.get("technical", {}).get("support_resistance")
            if support_resistance and (
                support_resistance.get("support_levels") or support_resistance.get("resistance_levels")
            ):
                symbol_logger.info(
                    f"Support/Resistance: {len(support_resistance.get('support_levels', []))} support, "
                    f"{len(support_resistance.get('resistance_levels', []))} resistance levels"
                )

            # Calculate multi-dimensional risk scores
            symbol_logger.info("Calculating multi-dimensional risk scores")
            risk_scores = self._calculate_risk_scores(symbol, quarterly_metrics, latest_data, multi_year_trends)
            if risk_scores:
                overall_risk = risk_scores.get("overall_risk", "N/A")
                risk_rating = risk_scores.get("risk_rating", "N/A")
                symbol_logger.info(
                    f"Risk Assessment: Overall {overall_risk}/10 ({risk_rating}) - "
                    f"Financial: {risk_scores.get('financial_health_risk', 'N/A')}, "
                    f"Market: {risk_scores.get('market_risk', 'N/A')}, "
                    f"Operational: {risk_scores.get('operational_risk', 'N/A')}"
                )

            # Fetch competitive positioning data
            symbol_logger.info("Fetching competitive positioning data")
            competitive_positioning = self._fetch_competitive_positioning_data(symbol, quarterly_metrics)
            if competitive_positioning and competitive_positioning.get("peers"):
                num_peers = len(competitive_positioning.get("peers", []))
                industry = competitive_positioning.get("industry", "Unknown")
                symbol_logger.info(f"Competitive Position: {num_peers} peers in {industry} industry")

            # Build peer performance leaderboard
            symbol_logger.info("Building peer performance leaderboard")
            peer_leaderboard = self._build_peer_performance_leaderboard(symbol, quarterly_metrics)
            if peer_leaderboard and peer_leaderboard.get("peers"):
                # Find target's rank
                target_peer = next((p for p in peer_leaderboard["peers"] if p["is_target"]), None)
                if target_peer:
                    overall_rank = target_peer.get("overall_rank", "N/A")
                    total_peers = peer_leaderboard.get("total_peers", "N/A")
                    symbol_logger.info(f"Peer Leaderboard: {symbol} ranks #{overall_rank} of {total_peers} peers")

            # Calculate volume profile
            symbol_logger.info("Calculating volume profile analysis")
            volume_profile = self._calculate_volume_profile(symbol, latest_data)
            if volume_profile and volume_profile.get("profile_bins"):
                poc_price = volume_profile.get("poc_price", "N/A")
                current_price = volume_profile.get("current_price", "N/A")
                days_analyzed = volume_profile.get("days_analyzed", "N/A")
                symbol_logger.info(
                    f"Volume Profile: POC at ${poc_price:.2f} (Current: ${current_price:.2f}), {days_analyzed} days analyzed"
                )

            # Tier 4: Monte Carlo Simulation for Probabilistic Forecasting
            monte_carlo_results = None
            try:
                symbol_logger.info("Running Monte Carlo simulation for probabilistic price forecasting")
                from utils.monte_carlo import MonteCarloSimulator

                # Get current price and volatility from latest data
                current_price_mc = latest_data.get("current_price", 0)
                if current_price_mc == 0:
                    current_price_mc = latest_data.get("technical", {}).get("current_price", 0)

                if current_price_mc > 0:
                    # Get volatility from technical indicators
                    technical_data = latest_data.get("technical", {})
                    volatility_annual = technical_data.get("volatility_annual", 0.25)  # Default 25%

                    # Run simulation
                    simulator = MonteCarloSimulator(random_seed=42)
                    monte_carlo_results = simulator.simulate_geometric_brownian_motion(
                        symbol=symbol,
                        current_price=current_price_mc,
                        volatility_annual=volatility_annual,
                        drift_annual=0.10,  # 10% expected return (can be adjusted based on historical data)
                        time_horizon_days=252,  # 1 year forecast
                        simulations=10000,
                    )

                    if monte_carlo_results:
                        mean_price = monte_carlo_results.mean_price
                        prob_profit = monte_carlo_results.probability_profit
                        var_95 = monte_carlo_results.var_95
                        symbol_logger.info(
                            f"Monte Carlo (1Y): Mean ${mean_price:.2f}, "
                            f"P(profit)={prob_profit:.1%}, VaR(95%)=${var_95:.2f}"
                        )

                        # Generate scenarios
                        scenarios = simulator.generate_scenarios(
                            symbol=symbol,
                            current_price=current_price_mc,
                            volatility_annual=volatility_annual,
                            drift_annual=0.10,
                            time_horizon_days=252,
                            simulations=10000,
                        )
                        monte_carlo_results.scenarios = scenarios
                else:
                    symbol_logger.warning("Cannot run Monte Carlo simulation - current price not available")

            except Exception as e:
                symbol_logger.error(f"Error running Monte Carlo simulation: {e}")
                monte_carlo_results = None

            # Tier 4: Chart Pattern Recognition
            chart_patterns = None
            try:
                symbol_logger.info("Detecting chart patterns in price data")
                import pandas as pd

                from utils.pattern_recognition import PatternRecognizer

                # Get historical price data (need OHLCV data for pattern recognition)
                technical_data = latest_data.get("technical", {})

                # Try to get price history from database
                price_history = self._fetch_price_history(symbol, days=252)

                if price_history and len(price_history) >= 30:
                    # Create DataFrame for pattern recognition
                    price_df = pd.DataFrame(price_history)

                    # Ensure required columns exist
                    if all(col in price_df.columns for col in ["date", "close", "high", "low", "open", "volume"]):
                        recognizer = PatternRecognizer()
                        detected_patterns = recognizer.detect_patterns(price_df)

                        chart_patterns = {
                            "patterns": detected_patterns,
                            "pattern_count": len(detected_patterns),
                            "days_analyzed": len(price_df),
                        }

                        if detected_patterns:
                            pattern_types = [p.pattern_type.value for p in detected_patterns]
                            bullish = sum(1 for p in detected_patterns if p.direction == "bullish")
                            bearish = sum(1 for p in detected_patterns if p.direction == "bearish")
                            symbol_logger.info(
                                f"Chart Patterns: {len(detected_patterns)} detected "
                                f"({bullish} bullish, {bearish} bearish) - {', '.join(pattern_types[:3])}"
                            )
                        else:
                            symbol_logger.info("Chart Patterns: No significant patterns detected")
                    else:
                        symbol_logger.warning(f"Chart patterns skipped - missing required columns in price data")
                else:
                    symbol_logger.warning(
                        f"Chart patterns skipped - insufficient price history (need 30+ days, have {len(price_history) if price_history else 0})"
                    )

            except Exception as e:
                symbol_logger.error(f"Error detecting chart patterns: {e}")
                chart_patterns = None

            # Calculate base scores
            symbol_logger.info("Calculating analysis scores")
            fundamental_score = self._calculate_fundamental_score(llm_responses)
            technical_score = self._calculate_technical_score(llm_responses)

            # Calculate weighted overall score
            overall_score = self._calculate_weighted_score(fundamental_score, technical_score)

            # Get current price from latest data (database) first, then try LLM response
            current_price = latest_data.get("current_price", 0)
            if current_price == 0:
                current_price = latest_data.get("technical", {}).get("current_price", 0)

            # If still no price, try to extract from technical LLM response
            if current_price == 0 and llm_responses.get("technical"):
                tech_content = llm_responses["technical"].get("content", "")
                if isinstance(tech_content, dict):
                    tech_content = json.dumps(tech_content)
                elif not isinstance(tech_content, str):
                    tech_content = str(tech_content)
                import re

                price_match = re.search(r'"current_price":\s*([\d.]+)', tech_content)
                if price_match:
                    current_price = float(price_match.group(1))

            # NEW APPROACH: Extract everything from existing LLM responses
            symbol_logger.info("=== NEW DIRECT EXTRACTION APPROACH ===")

            # Extract comprehensive SEC analysis data
            sec_data = self._extract_sec_comprehensive_data(llm_responses)
            if not isinstance(sec_data, dict):
                symbol_logger.warning(
                    f"SEC data extraction returned unexpected type: {type(sec_data)}, using empty dict"
                )
                sec_data = {}
            symbol_logger.info(
                f"SEC data extracted: business_quality={sec_data.get('business_quality_score', 0):.1f}, financial_health={sec_data.get('financial_health_score', 0):.1f}"
            )

            # Extract technical indicators
            try:
                tech_indicators = self._extract_technical_indicators(llm_responses)
                # Ensure tech_indicators is a dict, not a float or other type
                if not isinstance(tech_indicators, dict):
                    symbol_logger.warning(
                        f"Technical indicators extraction returned unexpected type: {type(tech_indicators)}, using empty dict"
                    )
                    tech_indicators = {}
                symbol_logger.info(
                    f"Technical indicators extracted: trend={tech_indicators.get('trend_direction', 'N/A')}, support_levels={len(tech_indicators.get('support_levels', []))}"
                )
            except Exception as e:
                symbol_logger.error(f"Error in technical indicators extraction: {e}")
                tech_indicators = {}

            # Calculate detailed data quality
            symbol_logger.info("Calculating detailed data quality")
            data_quality_detailed = self._calculate_data_quality_detailed(
                symbol, llm_responses, quarterly_metrics, latest_data
            )
            data_quality = data_quality_detailed["overall_score"] / 10  # Convert to 0-10 scale
            symbol_logger.info(
                f"Data quality: {data_quality_detailed['grade']} ({data_quality_detailed['overall_score']:.1f}%)"
            )

            # Check if we can use direct extraction optimization
            use_direct_extraction = sec_data and tech_indicators

            if use_direct_extraction:
                symbol_logger.info("OPTIMIZATION: Using direct LLM extraction - skipping traditional synthesis")
                # We'll create the recommendation directly later, for now just set a flag
                direct_extraction_data = {
                    "sec_data": sec_data,
                    "tech_indicators": tech_indicators,
                    "current_price": current_price,
                    "overall_score": overall_score,
                }
                # Set placeholder values for direct extraction
                synthesis_prompt = "Direct extraction from LLM analysis responses - no synthesis prompt needed"
                synthesis_response = ""  # Will be set later
                synthesis_metadata = {"source": "direct_extraction", "model": "comprehensive_analysis"}

            # Traditional synthesis path (or minimal fallback)
            symbol_logger.info("Preparing traditional synthesis path")

            # Use prompt manager for synthesis
            from investigator.application.prompts import get_prompt_manager

            prompt_manager = get_prompt_manager()

            # Prepare data for synthesis prompt
            from utils.synthesis_helpers import (
                format_fundamental_data_for_synthesis,
                format_technical_data_for_synthesis,
                get_performance_data,
            )

            fundamental_data_str = format_fundamental_data_for_synthesis(llm_responses.get("fundamental", {}))
            technical_data_str = format_technical_data_for_synthesis(llm_responses.get("technical", {}))

            # Get peer comparison data
            symbol_logger.info("Fetching peer comparison data")
            peer_analyzer = get_peer_comparison_analyzer()
            peer_comparison = peer_analyzer.get_peer_comparison(symbol)

            # Generate synthesis using LLM (but only if not using direct extraction)
            model_name = self.config.ollama.models.get("synthesis", "deepseek-r1:32b")

            if not use_direct_extraction:
                # Only run traditional synthesis if direct extraction failed
                # Generate synthesis using LLM
                model_name = self.config.ollama.models.get("synthesis", "deepseek-r1:32b")

                # Choose synthesis approach based on mode
                if synthesis_mode == "quarterly":
                    symbol_logger.info(f"Using quarterly synthesis mode (last N quarters + technical analysis)")
                    synthesis_prompt = self._create_quarterly_synthesis_prompt(
                        symbol, llm_responses, latest_data, prompt_manager
                    )
                else:
                    # Determine if we should use peer-enhanced synthesis
                    use_peer_synthesis = (
                        peer_comparison
                        and peer_comparison.get("company_ratios")
                        and peer_comparison.get("peer_statistics")
                        and len(peer_comparison.get("peer_statistics", {})) > 5  # At least 5 metrics
                    )

                    if use_peer_synthesis:
                        symbol_logger.info(
                            f"Using peer-enhanced synthesis with {peer_comparison.get('peers_analyzed', 0)} peers"
                        )
                        # Use peer-enhanced prompt
                        synthesis_prompt = prompt_manager.render_investment_synthesis_peer_prompt(
                            symbol=symbol,
                            analysis_date=datetime.now().strftime("%Y-%m-%d"),
                            current_price=latest_data.get("current_price", 0.0),
                            sector=peer_comparison.get("peer_group", {}).get("sector", "N/A"),
                            industry=peer_comparison.get("peer_group", {}).get("industry", "N/A"),
                            fundamental_data=fundamental_data_str,
                            technical_data=technical_data_str,
                            latest_market_data=str(latest_data),
                            peer_list=peer_comparison.get("peer_group", {}).get("peers", [])[:10],
                            company_ratios=peer_comparison.get("company_ratios", {}),
                            peer_statistics=peer_comparison.get("peer_statistics", {}),
                            relative_position=peer_comparison.get("relative_position", {}),
                        )
                    else:
                        symbol_logger.info("Using comprehensive synthesis mode")

                        # Debug: Log available fundamental keys
                        fundamental_keys = list(llm_responses.get("fundamental", {}).keys())
                        symbol_logger.info(f"Available fundamental keys: {fundamental_keys}")

                        # Extract comprehensive analysis and quarterly data
                        comprehensive_analysis = ""
                        quarterly_analyses = []

                        # Get comprehensive analysis if available
                        if "comprehensive" in llm_responses.get("fundamental", {}):
                            comp_data = llm_responses["fundamental"]["comprehensive"]
                            content = comp_data.get("content", comp_data)
                            symbol_logger.info(f"Found comprehensive analysis, content type: {type(content)}")
                            if isinstance(content, dict):
                                comprehensive_analysis = json.dumps(content, indent=2)
                                symbol_logger.info(f"Comprehensive analysis length: {len(comprehensive_analysis)}")
                            else:
                                comprehensive_analysis = str(content)
                        else:
                            symbol_logger.warning("No comprehensive analysis found in fundamental responses")

                        # Get quarterly analyses
                        for key, resp in llm_responses.get("fundamental", {}).items():
                            if key != "comprehensive":
                                qa = {
                                    "form_type": resp.get("form_type", "Unknown"),
                                    "period": resp.get("period", "Unknown"),
                                    "content": resp.get("content", {}),
                                }
                                quarterly_analyses.append(qa)

                        # Sort quarterly analyses by period
                        quarterly_analyses.sort(key=lambda x: x["period"])

                        # Use comprehensive synthesis template
                        synthesis_prompt = prompt_manager.render_investment_synthesis_comprehensive_prompt(
                            symbol=symbol,
                            analysis_date=datetime.now().strftime("%Y-%m-%d"),
                            current_price=latest_data.get("current_price", 0.0),
                            comprehensive_analysis=comprehensive_analysis,
                            quarterly_analyses=quarterly_analyses,
                            quarterly_count=len(quarterly_analyses),
                            technical_analysis=technical_data_str,
                            market_data=latest_data,
                        )

            # Enhanced system prompt for institutional-grade analysis
            system_prompt = """You are a senior portfolio manager and CFA charterholder with 25+ years of institutional investment experience. You excel at:

• Synthesizing complex multi-source financial analyses into actionable investment decisions
• Risk-adjusted portfolio construction for $2B+ institutional mandates
• Quantitative valuation analysis across market cycles and economic regimes
• Technical analysis integration with fundamental research for optimal timing
• ESG integration and fiduciary standard investment processes

Your responses must be precise, quantitative, and suitable for institutional investment committees. Focus on risk-adjusted returns, position sizing discipline, and clear execution frameworks. Provide specific price targets, stop-losses, and measurable investment criteria."""

            # Check cache first for synthesis response
            # Determine fiscal period for synthesis cache lookup
            fiscal_year, fiscal_period = self._get_latest_fiscal_period()

            # Use synthesis mode-specific cache key
            llm_type = f"synthesis_{synthesis_mode}"  # synthesis_comprehensive or synthesis_quarterly
            cache_key = {
                "symbol": symbol,
                "form_type": "SYNTHESIS",  # Use consistent intelligent default
                "period": f"{fiscal_year}-{fiscal_period}",
                "fiscal_year": fiscal_year,  # Separate key for file pattern
                "fiscal_period": fiscal_period,  # Separate key for file pattern
                "llm_type": llm_type,
            }

            cached_response = self.cache_manager.get(CacheType.LLM_RESPONSE, cache_key)

            if cached_response:
                symbol_logger.info(f"Using cached synthesis response for {symbol}")
                self.main_logger.info(f"Cache HIT for synthesis: {symbol}")

                # Use cached response directly (already processed by LLM facade)
                synthesis_response = cached_response.get("response", {})
                processing_time_ms = cached_response.get("metadata", {}).get("processing_time_ms", 0)
            else:
                symbol_logger.info(f"No cached synthesis found, generating with {model_name}")
                self.main_logger.info(f"Cache MISS for synthesis: {symbol}, generating with {model_name} (32K context)")

                start_time = time.time()
                # Use queue-based processing for synthesis
                from patterns.llm.llm_interfaces import LLMTaskType

                synthesis_task_data = {
                    "symbol": symbol,
                    "fundamental_data": fundamental_data_str,
                    "technical_data": technical_data_str,
                    "latest_data": latest_data,
                    "prompt": synthesis_prompt,
                }

                synthesis_result = self.ollama.generate_response(
                    task_type=LLMTaskType.SYNTHESIS, data=synthesis_task_data
                )

                processing_time_ms = int((time.time() - start_time) * 1000)

                # Use direct response from LLM facade (already processed)
                synthesis_response = synthesis_result

                # Save synthesis LLM response through cache manager
                self._save_synthesis_llm_response(
                    symbol, synthesis_prompt, synthesis_response, processing_time_ms, synthesis_mode
                )

            symbol_logger.info(f"Synthesis response generated in {processing_time_ms}ms")

            # Debug: Check what the synthesis response contains
            self.main_logger.info(f"Synthesis response type: {type(synthesis_response)}")
            self.main_logger.info(f"Synthesis response length: {len(str(synthesis_response))}")
            self.main_logger.info(f"Synthesis response preview: {str(synthesis_response)}")

            # Parse JSON synthesis response with metadata
            synthesis_metadata = {
                "model": model_name,
                "processing_time_ms": processing_time_ms,
                "symbol": symbol,
                "analysis_type": "investment_synthesis",
            }

            # DEBUG: Inspect incoming synthesis response data
            symbol_logger.info("=== SYNTHESIS RESPONSE DEBUG START ===")
            symbol_logger.info(f"synthesis_response type: {type(synthesis_response)}")
            symbol_logger.info(
                f"synthesis_response keys (if dict): {list(synthesis_response.keys()) if isinstance(synthesis_response, dict) else 'N/A'}"
            )

            if isinstance(synthesis_response, dict):
                symbol_logger.info(f"synthesis_response content preview: {str(synthesis_response)[:300]}...")
                # Check for common response formats
                if "content" in synthesis_response:
                    symbol_logger.info(f"Found 'content' key, type: {type(synthesis_response['content'])}")
                    symbol_logger.info(f"Content preview: {str(synthesis_response['content'])[:200]}...")
                if "response" in synthesis_response:
                    symbol_logger.info(f"Found 'response' key, type: {type(synthesis_response['response'])}")
                if "overall_score" in synthesis_response:
                    symbol_logger.info("Response appears to already be parsed JSON")
            else:
                symbol_logger.info(f"synthesis_response preview: {str(synthesis_response)[:300]}...")

            symbol_logger.info(f"synthesis_metadata: {synthesis_metadata}")
            symbol_logger.info("=== SYNTHESIS RESPONSE DEBUG END ===")

            # Robust JSON validation with fallback handling
            try:
                symbol_logger.info("BEFORE validation: Calling prompt_manager.validate_json_response")
                validated_response = prompt_manager.validate_json_response(synthesis_response, synthesis_metadata)
                symbol_logger.info(f"AFTER validation: validated_response type: {type(validated_response)}")
                symbol_logger.info(
                    f"AFTER validation: validated_response keys: {list(validated_response.keys()) if isinstance(validated_response, dict) else 'N/A'}"
                )

                # Check if validation failed
                if validated_response.get("error"):
                    symbol_logger.warning(f"JSON validation failed: {validated_response['error']}")
                    symbol_logger.warning(
                        f"Validation error details: {validated_response.get('details', 'No details')}"
                    )
                    # Try to extract any partial JSON or create fallback
                    ai_recommendation = self._create_fallback_recommendation(synthesis_response, symbol, overall_score)
                else:
                    ai_recommendation = validated_response["value"]
                    symbol_logger.info(f"Validation successful, ai_recommendation type: {type(ai_recommendation)}")

            except Exception as e:
                import traceback

                symbol_logger.error(f"EXCEPTION in JSON validation: {str(e)}")
                symbol_logger.error(f"Exception type: {type(e).__name__}")
                symbol_logger.error(f"Exception traceback: {traceback.format_exc()}")
                symbol_logger.error(f"Raw LLM response (first 500 chars): {str(synthesis_response)[:500]}")

                # Create a fallback recommendation based on computed scores
                ai_recommendation = self._create_fallback_recommendation(synthesis_response, symbol, overall_score)
                symbol_logger.info("Created fallback recommendation due to JSON parsing failure")

            # OPTIMIZATION: Use direct extraction if available
            if use_direct_extraction and "direct_extraction_data" in locals():
                symbol_logger.info("OPTIMIZATION: Overriding synthesis with direct extraction recommendation")
                ai_recommendation = self._create_recommendation_from_llm_data(
                    symbol,
                    direct_extraction_data["sec_data"],
                    direct_extraction_data["tech_indicators"],
                    direct_extraction_data["current_price"],
                    direct_extraction_data["overall_score"],
                )
                synthesis_response = json.dumps(ai_recommendation)
                synthesis_metadata = {"source": "direct_extraction", "model": "comprehensive_analysis"}
                symbol_logger.info("Direct extraction recommendation created successfully")

            # Handle different response types and capture additional insights
            additional_insights = []
            additional_risks = []

            # DEBUG: Inspect final ai_recommendation structure
            symbol_logger.info("=== AI_RECOMMENDATION DEBUG START ===")
            symbol_logger.info(f"ai_recommendation type: {type(ai_recommendation)}")
            symbol_logger.info(
                f"ai_recommendation keys: {list(ai_recommendation.keys()) if isinstance(ai_recommendation, dict) else 'N/A'}"
            )
            symbol_logger.info(f"ai_recommendation preview: {str(ai_recommendation)[:300]}...")

            thinking_content = ai_recommendation.get("thinking", "")
            symbol_logger.info(f"thinking_content length: {len(thinking_content)}")
            symbol_logger.info(
                f"thinking_content preview: {thinking_content[:100]}..." if thinking_content else "No thinking content"
            )

            additional_details = ai_recommendation.get("details", "")
            symbol_logger.info(f"additional_details length: {len(additional_details)}")
            symbol_logger.info(
                f"additional_details preview: {additional_details[:100]}..."
                if additional_details
                else "No additional details"
            )

            # Check for standard synthesis fields
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
            for field in standard_fields:
                if field in ai_recommendation:
                    field_value = ai_recommendation[field]
                    symbol_logger.info(f"Found {field}: {type(field_value)} = {str(field_value)[:100]}...")
                else:
                    symbol_logger.warning(f"Missing standard field: {field}")

            symbol_logger.info("=== AI_RECOMMENDATION DEBUG END ===")

            # Create extensible structure for additional insights and evolution
            extensible_insights = self._create_extensible_insights_structure(
                ai_recommendation, thinking_content, additional_details, symbol
            )
            symbol_logger.info(f"Created extensible insights structure with {len(extensible_insights)} sections")

            if additional_details:
                symbol_logger.info(f"Mixed response detected - capturing both JSON and additional text details")
                symbol_logger.info(f"Additional details captured: {len(additional_details)} characters")

            if thinking_content:
                symbol_logger.info(f"Captured {len(thinking_content)} chars of thinking/reasoning content")

            # Extract insights and risks from additional text details
            if additional_details:
                additional_insights, additional_risks = self._extract_insights_from_text(additional_details)
                symbol_logger.info(
                    f"Extracted {len(additional_insights)} insights and {len(additional_risks)} risks from text details"
                )

            # Also extract from thinking content if present
            if thinking_content:
                think_insights, think_risks = self._extract_insights_from_text(thinking_content)
                additional_insights.extend(think_insights)
                additional_risks.extend(think_risks)

            # Extract scores from parsed response or use defaults
            # Use LLM-provided scores if available
            overall_score = ai_recommendation.get("overall_score", overall_score)
            fundamental_score = ai_recommendation.get("fundamental_score", fundamental_score)
            technical_score = ai_recommendation.get("technical_score", technical_score)
            income_score = ai_recommendation.get(
                "income_statement_score", self._extract_income_score(llm_responses, ai_recommendation)
            )
            cashflow_score = ai_recommendation.get(
                "cash_flow_score", self._extract_cashflow_score(llm_responses, ai_recommendation)
            )
            balance_score = ai_recommendation.get(
                "balance_sheet_score", self._extract_balance_score(llm_responses, ai_recommendation)
            )
            growth_score = ai_recommendation.get(
                "growth_score", self._extract_growth_score(llm_responses, ai_recommendation)
            )
            value_score = ai_recommendation.get(
                "value_score", self._extract_value_score(llm_responses, ai_recommendation)
            )
            business_quality_score = ai_recommendation.get(
                "business_quality_score", self._extract_business_quality_score(llm_responses, ai_recommendation)
            )

            # Determine final recommendation with risk management
            final_recommendation = self._determine_final_recommendation(overall_score, ai_recommendation, data_quality)

            # Calculate price targets and risk levels
            price_target = self._calculate_price_target(symbol, llm_responses, ai_recommendation, current_price)
            stop_loss = self._calculate_stop_loss(current_price, final_recommendation, overall_score)

            # Clean symbol of any quotes that might have been added
            clean_symbol = symbol.strip("\"'")
            symbol_logger.info(f"Retrieved ai_recommendation from propmpt response.")

            # Create comprehensive recommendation
            recommendation = InvestmentRecommendation(
                symbol=clean_symbol,
                overall_score=overall_score,
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                income_score=income_score,
                cashflow_score=cashflow_score,
                balance_score=balance_score,
                growth_score=growth_score,
                value_score=value_score,
                business_quality_score=business_quality_score,
                recommendation=final_recommendation.get("recommendation", "HOLD"),
                confidence=final_recommendation.get("confidence", "LOW"),
                price_target=price_target,
                current_price=current_price,
                investment_thesis=ai_recommendation.get("executive_summary", {}).get(
                    "investment_thesis", ai_recommendation.get("investment_thesis", "Analysis based on available data")
                ),
                time_horizon=ai_recommendation.get("investment_recommendation", {}).get(
                    "time_horizon", ai_recommendation.get("time_horizon", "MEDIUM-TERM")
                ),
                position_size=self._extract_position_size(ai_recommendation),
                key_catalysts=self._extract_catalysts(ai_recommendation),
                key_risks=self._extract_comprehensive_risks(llm_responses, ai_recommendation, additional_risks),
                key_insights=self._extract_comprehensive_insights(
                    llm_responses, ai_recommendation, additional_insights
                ),
                entry_strategy=ai_recommendation.get("entry_strategy", ""),
                exit_strategy=ai_recommendation.get("exit_strategy", ""),
                stop_loss=stop_loss,
                analysis_timestamp=datetime.now(timezone.utc),
                data_quality_score=data_quality,
                analysis_thinking=ai_recommendation.get("analysis_thinking", thinking_content),
                synthesis_details=ai_recommendation.get("synthesis_details", additional_details),
                quarterly_metrics=quarterly_metrics,
                quarterly_trends=quarterly_trends,
                score_history=score_history,
                score_trend=score_trend,
                peer_valuation=peer_valuation,
                red_flags=red_flags,
                data_quality_detailed=data_quality_detailed,
                support_resistance=support_resistance,
                multi_year_trends=multi_year_trends,
                risk_scores=risk_scores,
                competitive_positioning=competitive_positioning,
                peer_leaderboard=peer_leaderboard,
                volume_profile=volume_profile,
                dcf_valuation=dcf_valuation,
                recession_performance=recession_performance,
                insider_trading=insider_trading,
                news_sentiment=news_sentiment,
                monte_carlo_results=monte_carlo_results,
                chart_patterns=chart_patterns,
            )

            # Attach extensible insights for enhanced reporting and future evolution
            recommendation.extensible_insights = extensible_insights
            symbol_logger.info(f"Attached extensible insights structure with {len(extensible_insights)} sections")

            # Evaluate alerts (Tier 3 Enhancement #14)
            if self.alert_engine:
                try:
                    symbol_logger.info("Evaluating alerts for significant changes")

                    # Get previous recommendation
                    previous_rec = self._get_previous_recommendation(symbol)

                    # Convert current recommendation to dict for alert evaluation
                    current_rec_dict = {
                        "symbol": recommendation.symbol,
                        "overall_score": recommendation.overall_score,
                        "fundamental_score": recommendation.fundamental_score,
                        "technical_score": recommendation.technical_score,
                        "recommendation": recommendation.recommendation,
                        "confidence": recommendation.confidence,
                        "current_price": recommendation.current_price,
                        "price_target": recommendation.price_target,
                        "support_resistance": recommendation.support_resistance,
                        "quarterly_metrics": recommendation.quarterly_metrics,
                    }

                    # Evaluate alerts
                    alerts = self.alert_engine.evaluate_alerts(
                        current_recommendation=current_rec_dict, previous_recommendation=previous_rec
                    )

                    if alerts:
                        symbol_logger.info(f"Generated {len(alerts)} alerts")

                        # Save alerts to database
                        for alert in alerts:
                            if alert["severity"] in ["high", "medium"]:
                                self.alert_engine.save_alert(alert)
                                symbol_logger.info(f"Saved {alert['severity']} severity alert: {alert['type']}")

                        # Send email for high severity alerts if configured
                        if self.email_notifier:
                            high_severity = [a for a in alerts if a["severity"] == "high"]
                            # Get recipients from email config
                            if high_severity and hasattr(self.config, "email") and self.config.email:
                                email_config = self.config.email
                                recipients = []
                                if hasattr(email_config, "recipients") and email_config.recipients:
                                    recipients = email_config.recipients

                                if recipients:
                                    try:
                                        for recipient in recipients:
                                            self.email_notifier.send_alert_email(
                                                recipient=recipient, alerts=high_severity
                                            )
                                        symbol_logger.info(
                                            f"Sent {len(high_severity)} high severity alerts to {len(recipients)} recipients"
                                        )
                                    except Exception as email_error:
                                        symbol_logger.error(f"Failed to send alert email: {email_error}")
                    else:
                        symbol_logger.info("No significant changes detected - no alerts generated")

                except Exception as alert_error:
                    symbol_logger.error(f"Error evaluating alerts: {alert_error}")
                    # Don't fail the synthesis if alert evaluation fails

            # Save synthesis results
            symbol_logger.info("Saving synthesis results to database")
            self._save_synthesis_results(symbol, recommendation)

            symbol_logger.info(
                f"Investment synthesis completed: {recommendation.recommendation} ({recommendation.overall_score:.1f}/10)"
            )
            self.main_logger.info(
                f"✅ Synthesis completed for {symbol}: {recommendation.recommendation} ({recommendation.overall_score:.1f}/10)"
            )

            # Track report generation in database
            processing_time = int((time.time() - start_time)) if "start_time" in locals() else None
            self._track_report_generation(recommendation, processing_time=processing_time)

            return recommendation

        except Exception as e:
            if "symbol_logger" in locals():
                symbol_logger.error(f"Investment synthesis failed: {str(e)}")
            self.main_logger.error(f"Error synthesizing analysis for {symbol}: {e}")
            import traceback

            self.main_logger.error(f"Traceback: {traceback.format_exc()}")
            # Raise the exception instead of returning a default recommendation
            raise RuntimeError(f"Investment synthesis failed for {symbol}: {str(e)}")

    def _analyze_recession_performance(self, symbol: str, multi_year_data: List[Dict]) -> Dict:
        """
        Analyze company performance during major economic crises

        Crisis Periods:
        - 2008 Financial Crisis: Q4 2007 - Q2 2009
        - 2020 COVID Crash: Q1 2020 - Q2 2020
        - Recovery periods: 2 years post-crisis

        Args:
            symbol: Stock symbol
            multi_year_data: Multi-year historical data

        Returns:
            Dictionary with crisis performance metrics
        """
        try:
            # Define crisis periods
            crises = {
                "financial_crisis_2008": {
                    "start": "2007-10-01",
                    "end": "2009-06-30",
                    "recovery_end": "2011-06-30",
                    "name": "2008 Financial Crisis",
                },
                "covid_2020": {
                    "start": "2020-01-01",
                    "end": "2020-06-30",
                    "recovery_end": "2022-06-30",
                    "name": "COVID-19 Pandemic",
                },
            }

            results = {}

            for crisis_id, crisis_info in crises.items():
                crisis_performance = self._calculate_crisis_metrics(
                    symbol, crisis_info["start"], crisis_info["end"], crisis_info["recovery_end"]
                )

                results[crisis_id] = {"name": crisis_info["name"], "performance": crisis_performance}

            # Calculate defensive score
            defensive_score = self._calculate_defensive_score(results)

            return {
                "crisis_performance": results,
                "defensive_score": defensive_score,
                "defensive_rating": self._get_defensive_rating(defensive_score),
            }

        except Exception as e:
            self.main_logger.error(f"Error analyzing recession performance for {symbol}: {e}")
            return {}

    def _calculate_crisis_metrics(self, symbol: str, start_date: str, end_date: str, recovery_end: str) -> Dict:
        """
        Calculate performance during and after crisis

        Args:
            symbol: Stock symbol
            start_date: Crisis start date
            end_date: Crisis end date
            recovery_end: End of recovery period

        Returns:
            Dictionary with crisis metrics
        """
        from sqlalchemy import text

        try:
            # Query SEC data for crisis period
            query = text(
                """
                SELECT sub.fy, sub.fp, sub.period,
                       num.tag, num.value
                FROM sec_sub_data sub
                JOIN sec_num_data num ON sub.adsh = num.adsh
                WHERE sub.cik = (SELECT cik FROM ticker_cik_mapping WHERE ticker = :symbol)
                    AND sub.period >= :start_date
                    AND sub.period <= :recovery_end
                    AND num.tag IN ('Revenues', 'NetIncomeLoss',
                                   'OperatingCashFlow', 'CashAndCashEquivalentsAtCarryingValue')
                ORDER BY sub.period
            """
            )

            with self.db_manager.get_session() as session:
                results = session.execute(
                    query, {"symbol": symbol, "start_date": start_date, "recovery_end": recovery_end}
                ).fetchall()

            # Parse into crisis vs recovery periods
            crisis_data = [r for r in results if r.period <= end_date]
            recovery_data = [r for r in results if r.period > end_date]

            # Calculate metrics
            metrics = {}

            # Revenue decline during crisis
            if crisis_data:
                crisis_revenues = [r.value for r in crisis_data if r.tag == "Revenues" and r.value]
                if len(crisis_revenues) >= 2:
                    revenue_decline = ((crisis_revenues[-1] / crisis_revenues[0]) - 1) * 100
                    metrics["revenue_decline_pct"] = round(revenue_decline, 2)

            # Earnings stability
            crisis_earnings = [r.value for r in crisis_data if r.tag == "NetIncomeLoss" and r.value]
            negative_quarters = sum(1 for e in crisis_earnings if e < 0)
            metrics["negative_earnings_quarters"] = negative_quarters
            if negative_quarters == 0:
                metrics["earnings_stability"] = "high"
            elif negative_quarters <= 1:
                metrics["earnings_stability"] = "medium"
            else:
                metrics["earnings_stability"] = "low"

            # Cash position maintained
            crisis_cash = [r.value for r in crisis_data if r.tag == "CashAndCashEquivalentsAtCarryingValue" and r.value]
            if len(crisis_cash) >= 2:
                cash_change = ((crisis_cash[-1] / crisis_cash[0]) - 1) * 100
                metrics["cash_position_change"] = round(cash_change, 2)

            # Recovery speed (quarters to pre-crisis revenue)
            if recovery_data and crisis_revenues:
                recovery_revenues = [r.value for r in recovery_data if r.tag == "Revenues" and r.value]
                if recovery_revenues:
                    pre_crisis_revenue = crisis_revenues[0]
                    quarters_to_recover = 0
                    for i, rev in enumerate(recovery_revenues):
                        if rev >= pre_crisis_revenue:
                            quarters_to_recover = i + 1
                            break
                    metrics["quarters_to_recover"] = quarters_to_recover if quarters_to_recover > 0 else "N/A"

            return metrics

        except Exception as e:
            self.main_logger.error(f"Error calculating crisis metrics for {symbol}: {e}")
            return {}

    def _calculate_defensive_score(self, crisis_results: Dict) -> float:
        """
        Calculate overall defensive characteristics score (0-10)

        High score = Stock holds up well in recessions

        Args:
            crisis_results: Dictionary of crisis performance data

        Returns:
            Defensive score 0-10
        """
        scores = []

        for crisis_id, crisis_data in crisis_results.items():
            perf = crisis_data.get("performance", {})

            # Score revenue stability (0-10)
            revenue_decline = perf.get("revenue_decline_pct", -50)
            if revenue_decline >= 0:
                revenue_score = 10
            elif revenue_decline >= -5:
                revenue_score = 8
            elif revenue_decline >= -10:
                revenue_score = 6
            elif revenue_decline >= -20:
                revenue_score = 4
            else:
                revenue_score = 2

            # Score earnings stability (0-10)
            earnings_stability = perf.get("earnings_stability", "low")
            if earnings_stability == "high":
                earnings_score = 10
            elif earnings_stability == "medium":
                earnings_score = 6
            else:
                earnings_score = 2

            # Score cash position (0-10)
            cash_change = perf.get("cash_position_change", -50)
            if cash_change >= 10:
                cash_score = 10
            elif cash_change >= 0:
                cash_score = 8
            elif cash_change >= -10:
                cash_score = 6
            else:
                cash_score = 3

            # Score recovery speed (0-10)
            quarters_to_recover = perf.get("quarters_to_recover", "N/A")
            if quarters_to_recover == "N/A":
                recovery_score = 5  # Neutral if unknown
            elif quarters_to_recover <= 2:
                recovery_score = 10
            elif quarters_to_recover <= 4:
                recovery_score = 7
            elif quarters_to_recover <= 8:
                recovery_score = 4
            else:
                recovery_score = 2

            # Weighted average for this crisis
            crisis_score = revenue_score * 0.3 + earnings_score * 0.3 + cash_score * 0.2 + recovery_score * 0.2
            scores.append(crisis_score)

        # Average across all crises
        return round(sum(scores) / len(scores), 1) if scores else 5.0

    def _get_defensive_rating(self, defensive_score: float) -> str:
        """
        Convert defensive score to rating

        Args:
            defensive_score: Score 0-10

        Returns:
            Rating string
        """
        if defensive_score >= 8.0:
            return "Very Defensive"
        elif defensive_score >= 6.5:
            return "Defensive"
        elif defensive_score >= 5.0:
            return "Neutral"
        elif defensive_score >= 3.0:
            return "Cyclical"
        else:
            return "Highly Cyclical"

    def _track_report_generation(
        self, recommendation: "InvestmentRecommendation", report_filename: str = None, processing_time: int = None
    ):
        """Track report generation in database for historical analysis"""
        try:
            # Get database connection
            db_config = self.config.database
            conn = psycopg2.connect(
                host=db_config.host,
                port=db_config.port,
                database=db_config.database,
                user=db_config.username,
                password=db_config.password,
            )
            cursor = conn.cursor()

            # Extract scores directly from recommendation attributes
            data = (
                recommendation.symbol,
                "synthesis",
                report_filename,
                recommendation.overall_score,
                recommendation.fundamental_score,
                recommendation.technical_score,
                recommendation.income_score,
                recommendation.balance_score,
                recommendation.cashflow_score,
                recommendation.growth_score,
                recommendation.value_score,
                recommendation.business_quality_score,
                recommendation.data_quality_score,
                recommendation.recommendation,
                recommendation.confidence,
                recommendation.time_horizon,
                recommendation.position_size,
                recommendation.current_price,
                recommendation.price_target,
                getattr(recommendation, "upside_potential", None),
                "comprehensive",
                "qwen2.5:32b-instruct-q4_K_M",  # Current model from config
                processing_time,
                True,  # sec_data_available
                True,  # technical_data_available
                8,  # quarters_analyzed (typical)
            )

            # Prepare data for insertion
            insert_sql = """
            INSERT INTO report_generation_history (
                ticker, report_type, report_filename,
                overall_score, fundamental_score, technical_score,
                income_statement_score, balance_sheet_score, cash_flow_score,
                growth_score, value_score, business_quality_score, data_quality_score,
                recommendation, confidence_level, time_horizon, position_size,
                current_price, target_price, upside_potential,
                analysis_mode, model_used, processing_time_seconds,
                sec_data_available, technical_data_available, 
                quarters_analyzed, market_date
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE
            )
            """

            cursor.execute(insert_sql, data)
            conn.commit()

            self.main_logger.info(f"📊 Report generation tracked for {recommendation.symbol}")

        except Exception as e:
            self.main_logger.error(f"Failed to track report generation for {recommendation.symbol}: {e}")
        finally:
            if "cursor" in locals():
                cursor.close()
            if "conn" in locals():
                conn.close()

    def generate_report(self, recommendations: List[InvestmentRecommendation], report_type: str = "synthesis") -> str:
        """
        Generate PDF report from recommendations

        Args:
            recommendations: List of investment recommendations
            report_type: Type of report to generate

        Returns:
            Path to generated report
        """
        try:
            # Convert recommendations to dict format for report generator
            rec_dicts = []
            for rec in recommendations:
                rec_dict = {
                    "symbol": rec.symbol,
                    "overall_score": rec.overall_score,
                    "fundamental_score": rec.fundamental_score,
                    "technical_score": rec.technical_score,
                    "income_score": rec.income_score,
                    "cashflow_score": rec.cashflow_score,
                    "balance_score": rec.balance_score,
                    "growth_score": rec.growth_score,
                    "value_score": rec.value_score,
                    "business_quality_score": rec.business_quality_score,
                    "data_quality_score": rec.data_quality_score,
                    "recommendation": rec.recommendation,
                    "confidence": rec.confidence,
                    "price_target": rec.price_target,
                    "current_price": rec.current_price,
                    "investment_thesis": rec.investment_thesis,
                    "time_horizon": rec.time_horizon,
                    "position_size": rec.position_size,
                    "key_catalysts": rec.key_catalysts,
                    "key_risks": rec.key_risks,
                    "key_insights": rec.key_insights,
                    "entry_strategy": rec.entry_strategy,
                    "exit_strategy": rec.exit_strategy,
                    "stop_loss": rec.stop_loss,
                    "data_quality_score": rec.data_quality_score,
                }
                rec_dicts.append(rec_dict)

            # Generate charts
            chart_paths = []

            # Generate technical charts for each symbol
            for rec in recommendations:
                # Load price data if available
                price_data_path = Path(self.config.data_dir) / "price_cache" / f"{rec.symbol}.parquet"
                if price_data_path.exists():
                    import pandas as pd

                    price_data = pd.read_parquet(price_data_path)
                    tech_chart = self.chart_generator.generate_technical_chart(
                        rec.symbol, price_data, rec.support_resistance
                    )
                    if tech_chart:
                        chart_paths.append(tech_chart)

                # Generate score history chart if data is available
                if rec.score_history and len(rec.score_history) >= 2:
                    self.main_logger.info(f"Generating score history chart for {rec.symbol}")
                    score_chart = self.chart_generator.generate_score_history_chart(
                        rec.symbol, rec.score_history, rec.score_trend or {}
                    )
                    if score_chart:
                        chart_paths.append(score_chart)

                # Generate valuation comparison chart if peer data is available
                if rec.peer_valuation:
                    self.main_logger.info(f"Generating valuation comparison chart for {rec.symbol}")
                    val_chart = self.chart_generator.generate_valuation_comparison_chart(rec.symbol, rec.peer_valuation)
                    if val_chart:
                        chart_paths.append(val_chart)

                # Generate quarterly trend charts if data is available
                if rec.quarterly_trends:
                    self.main_logger.info(f"Generating quarterly trend charts for {rec.symbol}")

                    # Revenue trend chart
                    revenue_chart = self.chart_generator.generate_quarterly_revenue_trend(
                        rec.symbol, rec.quarterly_trends
                    )
                    if revenue_chart:
                        chart_paths.append(revenue_chart)

                    # Profitability chart (Net Income + Margins)
                    profitability_chart = self.chart_generator.generate_quarterly_profitability_chart(
                        rec.symbol, rec.quarterly_trends
                    )
                    if profitability_chart:
                        chart_paths.append(profitability_chart)

                    # Cash Flow chart
                    cashflow_chart = self.chart_generator.generate_quarterly_cashflow_chart(
                        rec.symbol, rec.quarterly_trends
                    )
                    if cashflow_chart:
                        chart_paths.append(cashflow_chart)

                    self.main_logger.info(
                        f"Generated {len([c for c in [revenue_chart, profitability_chart, cashflow_chart] if c])} quarterly charts for {rec.symbol}"
                    )

                # Generate multi-year trends chart if data is available
                if rec.multi_year_trends:
                    self.main_logger.info(f"Generating multi-year historical trends chart for {rec.symbol}")
                    multi_year_chart = self.chart_generator.generate_multi_year_trends_chart(
                        rec.symbol, rec.multi_year_trends
                    )
                    if multi_year_chart:
                        chart_paths.append(multi_year_chart)
                        self.main_logger.info(f"Generated multi-year trends chart for {rec.symbol}")

                # Generate risk radar chart if data is available
                if rec.risk_scores:
                    self.main_logger.info(f"Generating risk radar chart for {rec.symbol}")
                    risk_radar_chart = self.chart_generator.generate_risk_scores_radar_chart(
                        rec.symbol, rec.risk_scores
                    )
                    if risk_radar_chart:
                        chart_paths.append(risk_radar_chart)
                        self.main_logger.info(f"Generated risk radar chart for {rec.symbol}")

                # Generate competitive positioning matrix if data is available
                if rec.competitive_positioning and rec.competitive_positioning.get("peers"):
                    self.main_logger.info(f"Generating competitive positioning matrix for {rec.symbol}")
                    positioning_chart = self.chart_generator.generate_competitive_positioning_matrix(
                        rec.competitive_positioning
                    )
                    if positioning_chart:
                        chart_paths.append(positioning_chart)
                        self.main_logger.info(f"Generated competitive positioning matrix for {rec.symbol}")

                # Generate volume profile chart if data is available
                if rec.volume_profile and rec.volume_profile.get("profile_bins"):
                    self.main_logger.info(f"Generating volume profile chart for {rec.symbol}")
                    volume_profile_chart = self.chart_generator.generate_volume_profile_chart(
                        rec.symbol, rec.volume_profile
                    )
                    if volume_profile_chart:
                        chart_paths.append(volume_profile_chart)
                        self.main_logger.info(f"Generated volume profile chart for {rec.symbol}")

            # Generate 3D fundamental plot for both single and multi-symbol reports
            # (Single symbol will show position in 3D space relative to ideal scores)
            fundamental_3d = self.chart_generator.generate_3d_fundamental_plot(rec_dicts)
            if fundamental_3d:
                chart_paths.append(fundamental_3d)

            # Generate 2D technical vs fundamental plot
            tech_fund_2d = self.chart_generator.generate_2d_technical_fundamental_plot(rec_dicts)
            if tech_fund_2d:
                chart_paths.append(tech_fund_2d)

            # Generate growth vs value plot
            growth_value_plot = self.chart_generator.generate_growth_value_plot(rec_dicts)
            if growth_value_plot:
                chart_paths.append(growth_value_plot)

            # Generate report based on type
            if report_type == "weekly":
                report_path = self.weekly_report_generator.generate_weekly_report(
                    portfolio_data=rec_dicts,
                    market_summary=self._get_market_summary(),
                    performance_data=self._calculate_portfolio_performance(rec_dicts),
                )
            else:
                report_path = self.report_generator.generate_report(
                    recommendations=rec_dicts, report_type=report_type, include_charts=chart_paths
                )

            self.main_logger.info(f"📊 Generated {report_type} report: {report_path}")
            return report_path

        except Exception as e:
            self.main_logger.error(f"Error generating report: {e}")
            raise

    def _fetch_llm_responses(self, symbol: str) -> Dict[str, Dict]:
        """Fetch ALL LLM responses for comprehensive synthesis including 8 quarterly + comprehensive + technical"""
        self.main_logger.info(f"Fetching LLM responses for {symbol}")

        try:
            llm_responses = {"fundamental": {}, "technical": None}

            # 1. FETCH COMPREHENSIVE FUNDAMENTAL ANALYSIS
            fiscal_year, fiscal_period = self._get_latest_fiscal_period()
            comp_cache_key = {
                "symbol": symbol,
                "form_type": "COMPREHENSIVE",
                "period": f"{fiscal_year}-FY",
                "llm_type": "sec",
            }

            comp_resp = self.cache_manager.get(CacheType.LLM_RESPONSE, comp_cache_key)
            if comp_resp:
                response_data = (
                    comp_resp.get("response", {})
                    if isinstance(comp_resp.get("response"), dict)
                    else comp_resp.get("response", {})
                )
                llm_responses["fundamental"]["comprehensive"] = {
                    "content": response_data,
                    "metadata": comp_resp.get("metadata", {}),
                    "form_type": "COMPREHENSIVE",
                    "period": f"{fiscal_year}-FY",
                }
                self.main_logger.info(f"✅ Fetched comprehensive fundamental analysis for {symbol}")

            # 2. FETCH ALL INDIVIDUAL QUARTERLY ANALYSES (for quarter-by-quarter trends)
            sec_responses = self.llm_dao.get_llm_responses_by_symbol(symbol, llm_type="sec")
            quarterly_count = 0

            for resp in sec_responses:
                form_type = resp["form_type"]
                period = resp["period"]

                # Skip comprehensive (already fetched above)
                if form_type == "COMPREHENSIVE":
                    continue

                # Parse period for cache key
                if "-" in period:
                    period_parts = period.split("-")
                    period_fiscal_year = int(period_parts[0])
                    period_fiscal_period = period_parts[1]
                else:
                    period_fiscal_year = fiscal_year
                    period_fiscal_period = "FY"

                cache_key = {"symbol": symbol, "form_type": form_type, "period": period, "llm_type": "sec"}

                cached_resp = self.cache_manager.get(CacheType.LLM_RESPONSE, cache_key)

                if cached_resp:
                    response_data = (
                        cached_resp.get("response", {})
                        if isinstance(cached_resp.get("response"), dict)
                        else cached_resp.get("response", {})
                    )

                    key = f"{form_type}_{period}"
                    llm_responses["fundamental"][key] = {
                        "content": response_data,
                        "metadata": cached_resp.get("metadata", {}),
                        "form_type": form_type,
                        "period": period,
                    }
                    quarterly_count += 1

            self.main_logger.info(f"✅ Fetched {quarterly_count} individual quarterly analyses for {symbol}")

            # 3. FETCH TECHNICAL ANALYSIS
            # Try multiple cache key formats for technical analysis
            technical_cache_keys = [
                {"symbol": symbol, "llm_type": "ta"},
                {"symbol": symbol, "llm_type": "technical"},
                {"symbol": symbol, "llm_type": "technical_analysis"},
                {"symbol": symbol, "form_type": "TECHNICAL", "llm_type": "ta"},
                {"symbol": symbol, "period": f"{fiscal_year}-Q1", "llm_type": "ta"},
            ]

            for tech_key in technical_cache_keys:
                tech_resp = self.cache_manager.get(CacheType.LLM_RESPONSE, tech_key)
                if tech_resp:
                    # Handle response that might be a JSON string with thinking content
                    raw_response = tech_resp.get("response", "")
                    if isinstance(raw_response, str):
                        # Check if it contains JSON (might have <think> prefix)
                        json_start = raw_response.find("{")
                        if json_start >= 0:
                            # Extract JSON part and parse it
                            json_content = raw_response[json_start:]
                            # Find the end by counting braces to handle nested JSON
                            brace_count = 0
                            json_end = 0
                            for i, char in enumerate(json_content):
                                if char == "{":
                                    brace_count += 1
                                elif char == "}":
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_end = i + 1
                                        break

                            if json_end > 0:
                                json_only = json_content[:json_end]
                                try:
                                    import json

                                    tech_content = json.loads(json_only)
                                except json.JSONDecodeError:
                                    tech_content = raw_response
                            else:
                                tech_content = raw_response
                        else:
                            tech_content = raw_response
                    else:
                        tech_content = raw_response if isinstance(raw_response, dict) else raw_response

                    llm_responses["technical"] = {
                        "content": tech_content,
                        "metadata": tech_resp.get("metadata", {}),
                        "cache_key_used": tech_key,
                    }
                    self.main_logger.info(f"✅ Fetched technical analysis for {symbol} using key: {tech_key}")
                    break

            # 4. FALLBACK: Check file-based technical analysis cache
            if not llm_responses["technical"]:
                try:
                    tech_file_path = f"data/llm_cache/{symbol}/response_technical_indicators.txt"
                    if Path(tech_file_path).exists():
                        with open(tech_file_path, "r") as f:
                            tech_content = f.read()
                        llm_responses["technical"] = {
                            "content": tech_content,
                            "metadata": {"source": "file_fallback"},
                            "cache_key_used": "file_fallback",
                        }
                        self.main_logger.info(f"✅ Fetched technical analysis from file for {symbol}")
                except Exception as e:
                    self.main_logger.warning(f"Failed to fetch technical analysis from file: {e}")

            if not llm_responses["technical"]:
                self.main_logger.warning(f"❌ No technical analysis found for {symbol}")
                llm_responses["technical"] = None

            # SUMMARY LOG
            fundamental_count = len(llm_responses["fundamental"])
            technical_count = 1 if llm_responses["technical"] else 0
            self.main_logger.info(
                f"📊 SYNTHESIS DATA SUMMARY: Retrieved {fundamental_count} fundamental analyses "
                f"(including {'✅ comprehensive + ' if 'comprehensive' in llm_responses['fundamental'] else '❌ no comprehensive, '}"
                f"{fundamental_count-1 if 'comprehensive' in llm_responses['fundamental'] else fundamental_count} quarterly) "
                f"and {technical_count} technical analysis for {symbol}"
            )

            return llm_responses

        except Exception as e:
            self.main_logger.error(f"Error fetching LLM responses: {e}")
            return {"fundamental": {}, "technical": None}

    def _fetch_latest_data(self, symbol: str) -> Dict:
        """Fetch latest fundamental and technical data from parquet files"""
        self.main_logger.info(f"Fetching latest data for {symbol}")

        try:
            # Fetch latest technical data from parquet file (preferred) or CSV fallback
            technical_data = {}

            # First try parquet format (compressed, efficient)
            parquet_data_path = (
                Path(self.config.data_dir) / "technical_cache" / symbol / f"technical_data_{symbol}.parquet"
            )
            csv_data_path = Path(self.config.data_dir) / "technical_cache" / symbol / f"technical_data_{symbol}.csv"

            self.main_logger.info(f"Looking for technical data - Parquet: {parquet_data_path}, CSV: {csv_data_path}")

            # Try parquet first, then CSV fallback
            technical_data_path = None
            if parquet_data_path.exists():
                technical_data_path = parquet_data_path
                read_func = lambda path: pd.read_parquet(path)
                file_type = "parquet"
            elif csv_data_path.exists():
                technical_data_path = csv_data_path
                read_func = lambda path: pd.read_csv(path)
                file_type = "CSV"

            if technical_data_path:
                import pandas as pd

                try:
                    # Read data file with technical analysis data
                    df = read_func(technical_data_path)
                    self.main_logger.info(
                        f"Successfully read {file_type} file with {len(df)} rows from {technical_data_path}"
                    )

                    if not df.empty:
                        # Get the latest row (most recent data)
                        latest_row = df.iloc[-1]

                        # Extract technical data from the latest row
                        # Handle volume which may be comma-formatted string (CSV) or proper numeric (parquet)
                        volume_raw = latest_row.get("Volume", 0)
                        if isinstance(volume_raw, str):
                            volume = int(volume_raw.replace(",", ""))
                        else:
                            volume = int(volume_raw)

                        technical_data = {
                            "current_price": float(latest_row.get("Close", 0)),
                            "price_change_1d": float(
                                latest_row.get("Price_Change_1D", 0)
                            ),  # Use the correct column name
                            "price_change_1w": float(latest_row.get("price_change_1w", 0)),
                            "price_change_1m": float(latest_row.get("price_change_1m", 0)),
                            "rsi": float(latest_row.get("RSI_14", 50)),
                            "macd": float(latest_row.get("MACD", 0)),
                            "sma_20": float(latest_row.get("SMA_20", 0)),
                            "sma_50": float(latest_row.get("SMA_50", 0)),
                            "sma_200": float(latest_row.get("SMA_200", 0)),
                            "volume": volume,
                            "analysis_date": str(latest_row.get("Date", "Unknown")),
                        }

                        # Detect support/resistance levels from full price history
                        try:
                            from utils.support_resistance import detect_support_resistance_levels

                            # Prepare DataFrame with required columns
                            price_df = df.copy()
                            # Standardize column names (handle both uppercase and lowercase)
                            if "Close" in price_df.columns:
                                price_df["close"] = price_df["Close"]
                            if "High" in price_df.columns:
                                price_df["high"] = price_df["High"]
                            if "Low" in price_df.columns:
                                price_df["low"] = price_df["Low"]

                            # Detect levels
                            sr_levels = detect_support_resistance_levels(price_df)
                            technical_data["support_resistance"] = sr_levels

                            self.main_logger.info(
                                f"Detected {len(sr_levels.get('support_levels', []))} support and "
                                f"{len(sr_levels.get('resistance_levels', []))} resistance levels for {symbol}"
                            )
                        except Exception as e:
                            self.main_logger.warning(f"Could not detect support/resistance levels for {symbol}: {e}")
                            technical_data["support_resistance"] = None

                        self.main_logger.info(
                            f"Loaded technical data from {file_type}: current_price=${technical_data['current_price']:.2f}"
                        )
                    else:
                        self.main_logger.warning(f"Empty {file_type} file for {symbol}")

                except Exception as e:
                    self.main_logger.error(f"Error reading CSV file for {symbol}: {e}")
            else:
                self.main_logger.warning(f"CSV file not found for {symbol}: {technical_data_path}")

            return {
                "fundamental": {},
                "technical": technical_data,
                "current_price": technical_data.get("current_price", 0),
            }

        except Exception as e:
            self.main_logger.error(f"Error fetching latest data for {symbol}: {e}")
            # Fail immediately as requested - no fallbacks that give wrong answers
            raise RuntimeError(f"Failed to fetch latest data for {symbol}: {e}")

    def _fetch_historical_scores(self, symbol: str) -> List[Dict]:
        """
        Fetch historical investment scores from synthesis_results table

        Args:
            symbol: Stock symbol

        Returns:
            List of score history dictionaries with date, score, recommendation
        """
        try:
            # Try to fetch from synthesis_results table (will be empty on first run)
            query = text(
                """
                SELECT
                    analysis_timestamp as generated_at,
                    overall_score,
                    fundamental_score,
                    technical_score,
                    recommendation,
                    confidence
                FROM synthesis_results
                WHERE symbol = :symbol
                ORDER BY generated_at ASC
            """
            )

            with self.db_manager.get_session() as session:
                result = session.execute(query, {"symbol": symbol})
                rows = result.fetchall()

            if not rows:
                self.main_logger.info(f"No historical scores found for {symbol}")
                return []

            score_history = []
            for row in rows:
                score_history.append(
                    {
                        "date": row[0] if hasattr(row, "__getitem__") else row.generated_at,
                        "overall_score": row[1] if hasattr(row, "__getitem__") else row.overall_score,
                        "fundamental_score": row[2] if hasattr(row, "__getitem__") else row.fundamental_score,
                        "technical_score": row[3] if hasattr(row, "__getitem__") else row.technical_score,
                        "recommendation": row[4] if hasattr(row, "__getitem__") else row.recommendation,
                        "confidence": row[5] if hasattr(row, "__getitem__") else row.confidence,
                    }
                )

            self.main_logger.info(f"Retrieved {len(score_history)} historical scores for {symbol}")
            return score_history

        except Exception as e:
            self.main_logger.error(f"Error fetching historical scores for {symbol}: {e}")
            return []

    def _fetch_peer_valuation_metrics(self, symbol: str) -> Dict:
        """
        Fetch valuation metrics for symbol and its peers

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with peer valuation data
        """
        try:
            # First, get the industry for this symbol
            industry_query = text(
                """
                SELECT industry, sector
                FROM peer_metrics
                WHERE symbol = :symbol
                LIMIT 1
            """
            )

            with self.db_manager.get_session() as session:
                result = session.execute(industry_query, {"symbol": symbol}).fetchone()

                if not result:
                    self.main_logger.warning(f"No peer group found for {symbol}")
                    return {}

                industry = result[0] if hasattr(result, "__getitem__") else result.industry
                sector = result[1] if hasattr(result, "__getitem__") else result.sector

                # Get peer metrics
                peer_query = text(
                    """
                    SELECT symbol, metrics_data
                    FROM peer_metrics
                    WHERE industry = :industry
                    ORDER BY symbol
                """
                )

                peer_result = session.execute(peer_query, {"industry": industry})
                peers = peer_result.fetchall()

            if not peers:
                return {}

            # Extract valuation metrics
            peer_valuations = []
            target_valuation = None

            for row in peers:
                sym = row[0] if hasattr(row, "__getitem__") else row.symbol
                metrics = dict(row[1]) if hasattr(row, "__getitem__") else row.metrics_data

                valuation = {
                    "symbol": sym,
                    "pe_ratio": metrics.get("pe_ratio"),
                    "pb_ratio": metrics.get("pb_ratio"),
                    "ps_ratio": metrics.get("ps_ratio"),
                    "peg_ratio": metrics.get("peg_ratio"),
                    "market_cap": metrics.get("market_cap"),
                    "revenue": metrics.get("revenue"),
                    "net_income": metrics.get("net_income"),
                }

                if sym == symbol:
                    target_valuation = valuation
                else:
                    peer_valuations.append(valuation)

            if not target_valuation:
                return {}

            # Calculate peer medians
            def safe_median(values):
                filtered = [v for v in values if v is not None and v > 0]
                return float(np.median(filtered)) if filtered else None

            peer_medians = {
                "pe_ratio": safe_median([p["pe_ratio"] for p in peer_valuations]),
                "pb_ratio": safe_median([p["pb_ratio"] for p in peer_valuations]),
                "ps_ratio": safe_median([p["ps_ratio"] for p in peer_valuations]),
                "peg_ratio": safe_median([p["peg_ratio"] for p in peer_valuations]),
            }

            return {
                "symbol": symbol,
                "industry": industry,
                "sector": sector,
                "target": target_valuation,
                "peer_medians": peer_medians,
                "peer_count": len(peer_valuations),
                "relative_valuation": self._calculate_relative_valuation(target_valuation, peer_medians),
            }

        except Exception as e:
            self.main_logger.error(f"Error fetching peer valuation metrics: {e}")
            return {}

    def _calculate_relative_valuation(self, target: Dict, peer_medians: Dict) -> Dict:
        """
        Calculate relative valuation metrics

        Args:
            target: Target company valuation metrics
            peer_medians: Peer group median metrics

        Returns:
            Dictionary with relative valuation analysis
        """
        try:
            relative = {}

            # Calculate premium/discount for each metric
            for metric in ["pe_ratio", "pb_ratio", "ps_ratio", "peg_ratio"]:
                target_val = target.get(metric)
                peer_median = peer_medians.get(metric)

                if target_val and peer_median and peer_median > 0:
                    premium = ((target_val - peer_median) / peer_median) * 100
                    relative[metric] = {
                        "target": round(target_val, 2),
                        "peer_median": round(peer_median, 2),
                        "premium_discount": round(premium, 1),
                        "assessment": "expensive" if premium > 20 else ("cheap" if premium < -20 else "fair"),
                    }
                else:
                    relative[metric] = {"target": target_val, "peer_median": peer_median, "premium_discount": None}

            # Overall valuation assessment
            assessments = [v.get("assessment") for v in relative.values() if v.get("assessment")]
            if assessments:
                cheap_count = assessments.count("cheap")
                expensive_count = assessments.count("expensive")

                if cheap_count > expensive_count:
                    overall = "undervalued"
                elif expensive_count > cheap_count:
                    overall = "overvalued"
                else:
                    overall = "fairly_valued"
            else:
                overall = "insufficient_data"

            relative["overall_assessment"] = overall

            return relative

        except Exception as e:
            self.main_logger.error(f"Error calculating relative valuation: {e}")
            return {}

    def _calculate_score_trend(self, score_history: List[Dict]) -> Dict:
        """
        Analyze score trend to determine if improving or declining

        Args:
            score_history: List of historical scores

        Returns:
            Dictionary with trend analysis
        """
        if len(score_history) < 2:
            return {"trend": "insufficient_data", "change": 0, "direction": "neutral"}

        try:
            scores = [s["overall_score"] for s in score_history]

            # Calculate trend
            first_score = scores[0]
            last_score = scores[-1]
            change = last_score - first_score

            # Calculate average of first half vs second half
            mid_point = len(scores) // 2
            first_half_avg = sum(scores[:mid_point]) / mid_point if mid_point > 0 else first_score
            second_half_avg = sum(scores[mid_point:]) / (len(scores) - mid_point)

            # Determine trend direction
            if change > 0.5:
                direction = "improving"
            elif change < -0.5:
                direction = "declining"
            else:
                direction = "stable"

            # Calculate momentum (recent trend)
            if len(scores) >= 3:
                recent_change = scores[-1] - scores[-3]
                momentum = "accelerating" if abs(recent_change) > abs(change) / 2 else "steady"
            else:
                momentum = "steady"

            return {
                "trend": direction,
                "change": round(change, 2),
                "change_percent": round((change / first_score) * 100, 1) if first_score > 0 else 0,
                "first_half_avg": round(first_half_avg, 2),
                "second_half_avg": round(second_half_avg, 2),
                "momentum": momentum,
                "num_analyses": len(scores),
            }

        except Exception as e:
            self.main_logger.error(f"Error calculating score trend: {e}")
            return {"trend": "error", "change": 0, "direction": "neutral"}

    def _detect_red_flags(
        self, symbol: str, quarterly_metrics: List[Dict], quarterly_trends: Dict, latest_data: Dict
    ) -> List[Dict]:
        """
        Detect warning signs in financial data

        Args:
            symbol: Stock symbol
            quarterly_metrics: List of quarterly metrics
            quarterly_trends: Dictionary of quarterly trends
            latest_data: Latest market data

        Returns:
            List of red flags with severity (high/medium/low)
        """
        red_flags = []

        if not quarterly_metrics or len(quarterly_metrics) < 2:
            return red_flags

        # RED FLAG #1: Declining Revenue (2+ quarters)
        revenue_trend = quarterly_trends.get("revenue_trend", [])
        if len(revenue_trend) >= 3:
            last_3 = revenue_trend[-3:]
            # Check if all values are non-None before comparing
            if all(item.get("value") is not None for item in last_3):
                if all(last_3[i]["value"] < last_3[i - 1]["value"] for i in range(1, 3)):
                    red_flags.append(
                        {
                            "type": "declining_revenue",
                            "severity": "high",
                            "description": "Revenue declining for 2+ consecutive quarters",
                            "detail": f"Revenue dropped from ${last_3[0]['value']:.0f}M to ${last_3[-1]['value']:.0f}M",
                        }
                    )

        # RED FLAG #2: Negative cash flow with positive earnings
        latest_quarter = quarterly_metrics[-1] if quarterly_metrics else {}
        net_income = latest_quarter.get("net_income", 0) or 0
        ocf = latest_quarter.get("operating_cash_flow", 0) or 0

        if net_income and net_income > 0 and ocf is not None and ocf < 0:
            red_flags.append(
                {
                    "type": "negative_cash_flow",
                    "severity": "high",
                    "description": "Negative operating cash flow despite positive earnings",
                    "detail": f"Net Income: ${net_income/1e6:.0f}M, OCF: ${ocf/1e6:.0f}M (earnings quality concern)",
                }
            )

        # RED FLAG #3: Debt/Equity spike
        if len(quarterly_metrics) >= 4:
            latest = quarterly_metrics[-1]
            four_quarters_ago = quarterly_metrics[-4]

            latest_de = latest.get("debt_to_equity", 0)
            prev_de = four_quarters_ago.get("debt_to_equity", 0)

            if latest_de and prev_de and latest_de > prev_de * 1.5:
                red_flags.append(
                    {
                        "type": "debt_spike",
                        "severity": "medium",
                        "description": "Debt/Equity ratio increased significantly",
                        "detail": f"D/E increased {((latest_de/prev_de - 1) * 100):.0f}% over past year",
                    }
                )

        # RED FLAG #4: Margin compression
        margin_trends = quarterly_trends.get("margin_trends", [])
        if len(margin_trends) >= 4:
            # Handle None values in net_margin
            recent_margins = [m.get("net_margin", 0) or 0 for m in margin_trends[-2:]]
            past_margins = [m.get("net_margin", 0) or 0 for m in margin_trends[:2]]
            recent_avg = sum(recent_margins) / len(recent_margins) if recent_margins else 0
            past_avg = sum(past_margins) / len(past_margins) if past_margins else 0

            if past_avg and past_avg > 0 and recent_avg and recent_avg < past_avg * 0.8:
                red_flags.append(
                    {
                        "type": "margin_compression",
                        "severity": "medium",
                        "description": "Net margins compressing significantly",
                        "detail": f"Margins dropped from {past_avg:.1f}% to {recent_avg:.1f}%",
                    }
                )

        # RED FLAG #5: Accounts receivable days increasing
        if len(quarterly_metrics) >= 4:
            latest = quarterly_metrics[-1]
            four_quarters_ago = quarterly_metrics[-4]

            latest_ar_days = latest.get("accounts_receivable_days", 0)
            prev_ar_days = four_quarters_ago.get("accounts_receivable_days", 0)

            if latest_ar_days and prev_ar_days and latest_ar_days > prev_ar_days * 1.3:
                red_flags.append(
                    {
                        "type": "collection_issues",
                        "severity": "medium",
                        "description": "Accounts receivable days increasing (collection problems)",
                        "detail": f"AR days increased from {prev_ar_days:.0f} to {latest_ar_days:.0f}",
                    }
                )

        return red_flags

    def _fetch_multi_year_financials(self, symbol: str, years: int = 5) -> List[Dict]:
        """
        Fetch multi-year annual financial data from SEC tables

        Args:
            symbol: Stock symbol
            years: Number of years to fetch (default: 10)

        Returns:
            List of yearly financial data dictionaries
        """
        try:
            # Get CIK for this symbol
            cik_query = text("SELECT cik FROM ticker_cik_mapping WHERE ticker = :symbol")

            with self.db_manager.get_session() as session:
                cik_result = session.execute(cik_query, {"symbol": symbol}).fetchone()

                if not cik_result:
                    self.main_logger.warning(f"No CIK found for {symbol}")
                    return []

                cik = int(cik_result[0])

            # Calculate year range
            current_year = datetime.now().year
            start_year = current_year - years

            # Query annual data from SEC tables
            query = text(
                """
                SELECT
                    sub.fy as fiscal_year,
                    sub.fp as fiscal_period,
                    num.tag,
                    num.value,
                    num.uom as units
                FROM sec_sub_data sub
                JOIN sec_num_data num ON sub.adsh = num.adsh
                WHERE
                    sub.cik = :cik
                    AND sub.fy >= :start_year
                    AND sub.fy <= :end_year
                    AND sub.fp = 'FY'
                    AND num.qtrs = 4
                    AND num.tag IN (
                        'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax',
                        'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax',
                        'NetIncomeLoss', 'EarningsPerShareBasic',
                        'Assets', 'Liabilities', 'StockholdersEquity',
                        'OperatingIncomeLoss', 'CashAndCashEquivalentsAtCarryingValue',
                        'GrossProfit', 'OperatingExpenses'
                    )
                ORDER BY sub.fy DESC
            """
            )

            with self.db_manager.get_session() as session:
                result = session.execute(query, {"cik": cik, "start_year": start_year, "end_year": current_year})
                rows = result.fetchall()

            if not rows:
                self.main_logger.warning(f"No multi-year data found for {symbol}")
                return []

            # Structure the data by year
            yearly_data = self._structure_yearly_sec_data(rows)

            self.main_logger.info(f"Fetched {len(yearly_data)} years of financial data for {symbol}")
            return yearly_data

        except Exception as e:
            self.main_logger.error(f"Error fetching multi-year financials for {symbol}: {e}")
            return []

    def _structure_yearly_sec_data(self, rows: List) -> List[Dict]:
        """Structure raw SEC data into yearly metrics"""
        yearly_data = {}

        for row in rows:
            year = row.fiscal_year
            tag = row.tag
            value = row.value

            if year not in yearly_data:
                yearly_data[year] = {
                    "year": year,
                    "revenue": None,
                    "net_income": None,
                    "assets": None,
                    "equity": None,
                    "operating_income": None,
                    "gross_profit": None,
                }

            # Map SEC tags to metrics
            if tag in [
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
            ]:
                if yearly_data[year]["revenue"] is None:
                    yearly_data[year]["revenue"] = float(value) if value else None

            elif tag == "NetIncomeLoss":
                yearly_data[year]["net_income"] = float(value) if value else None

            elif tag == "Assets":
                yearly_data[year]["assets"] = float(value) if value else None

            elif tag == "StockholdersEquity":
                yearly_data[year]["equity"] = float(value) if value else None

            elif tag == "OperatingIncomeLoss":
                yearly_data[year]["operating_income"] = float(value) if value else None

            elif tag == "GrossProfit":
                yearly_data[year]["gross_profit"] = float(value) if value else None

        # Convert to sorted list
        return sorted(yearly_data.values(), key=lambda x: x["year"])

    def _calculate_multi_year_metrics(self, yearly_data: List[Dict]) -> Dict:
        """
        Calculate CAGR, trends, and patterns from multi-year data

        Returns:
            Dictionary with calculated metrics
        """
        if len(yearly_data) < 3:
            return {"error": "Insufficient data for multi-year analysis", "years_available": len(yearly_data)}

        metrics = {"years_analyzed": len(yearly_data)}

        # Revenue CAGR
        revenue_values = [(y["year"], y["revenue"]) for y in yearly_data if y["revenue"] and y["revenue"] > 0]
        if len(revenue_values) >= 3:
            first_year, first_revenue = revenue_values[0]
            last_year, last_revenue = revenue_values[-1]
            years_span = last_year - first_year

            if years_span > 0 and first_revenue > 0:
                revenue_cagr = ((last_revenue / first_revenue) ** (1 / years_span) - 1) * 100
                metrics["revenue_cagr"] = round(revenue_cagr, 2)
                metrics["revenue_first"] = first_revenue
                metrics["revenue_last"] = last_revenue

        # Earnings CAGR
        earnings_values = [(y["year"], y["net_income"]) for y in yearly_data if y["net_income"] and y["net_income"] > 0]
        if len(earnings_values) >= 3:
            first_year, first_earnings = earnings_values[0]
            last_year, last_earnings = earnings_values[-1]
            years_span = last_year - first_year

            if years_span > 0 and first_earnings > 0:
                earnings_cagr = ((last_earnings / first_earnings) ** (1 / years_span) - 1) * 100
                metrics["earnings_cagr"] = round(earnings_cagr, 2)

        # Revenue volatility
        if len(revenue_values) >= 3:
            growth_rates = []
            for i in range(1, len(revenue_values)):
                prev_revenue = revenue_values[i - 1][1]
                curr_revenue = revenue_values[i][1]
                if prev_revenue > 0:
                    growth_rate = (curr_revenue / prev_revenue - 1) * 100
                    growth_rates.append(growth_rate)

            if growth_rates:
                import numpy as np

                metrics["revenue_volatility"] = round(np.std(growth_rates), 2)
                metrics["avg_growth_rate"] = round(np.mean(growth_rates), 2)

        # Trend direction
        revenues_only = [y["revenue"] for y in yearly_data if y["revenue"]]
        if len(revenues_only) >= 3:
            from numpy.polynomial import Polynomial

            p = Polynomial.fit(range(len(revenues_only)), revenues_only, 1)
            slope = p.coef[1]

            if slope > 0:
                metrics["trend_direction"] = "growing"
            elif slope < 0:
                metrics["trend_direction"] = "declining"
            else:
                metrics["trend_direction"] = "stable"

        # Cyclical pattern
        metrics["cyclical_pattern"] = self._detect_cyclical_pattern(yearly_data)

        return metrics

    def _detect_cyclical_pattern(self, yearly_data: List[Dict]) -> str:
        """Detect if business shows cyclical pattern"""
        revenue_values = [y["revenue"] for y in yearly_data if y["revenue"] and y["revenue"] > 0]

        if len(revenue_values) < 3:
            return "insufficient_data"

        growth_rates = []
        for i in range(1, len(revenue_values)):
            if revenue_values[i - 1] > 0:
                growth_rate = (revenue_values[i] / revenue_values[i - 1] - 1) * 100
                growth_rates.append(growth_rate)

        if growth_rates:
            import numpy as np

            volatility = np.std(growth_rates)

            if volatility > 15:
                return "highly_cyclical"
            elif volatility > 8:
                return "moderately_cyclical"
            else:
                return "stable"

        return "unknown"

    def _calculate_risk_scores(
        self, symbol: str, quarterly_metrics: List[Dict], latest_data: Dict, multi_year_trends: Optional[Dict]
    ) -> Dict:
        """
        Calculate multi-dimensional risk scores across 5 dimensions

        Args:
            symbol: Stock symbol
            quarterly_metrics: List of quarterly financial metrics
            latest_data: Latest technical analysis data
            multi_year_trends: Multi-year historical trends data

        Returns:
            Dictionary with dimension scores (0-10, lower is better) and overall risk rating
        """
        import numpy as np

        risk_scores = {}

        if not quarterly_metrics or len(quarterly_metrics) == 0:
            return {
                "overall_risk": 10.0,
                "risk_rating": "Very High",
                "error": "No data available",
                "financial_health_risk": 10.0,
                "market_risk": 10.0,
                "operational_risk": 10.0,
                "business_model_risk": 10.0,
                "growth_risk": 10.0,
            }

        latest = quarterly_metrics[-1] if quarterly_metrics else {}

        # 1. Financial Health Risk (0-10, lower is better)
        # Based on leverage, liquidity, solvency
        debt_to_equity = latest.get("debt_to_equity", 0) or 0
        current_ratio = latest.get("current_ratio", 0) or 0

        financial_health_risk = 5.0  # Default medium risk
        if debt_to_equity > 2.0:
            financial_health_risk += 2.0
        elif debt_to_equity > 1.0:
            financial_health_risk += 0.5
        elif debt_to_equity < 0.5:
            financial_health_risk -= 1.5

        if current_ratio < 1.0:
            financial_health_risk += 2.0
        elif current_ratio > 2.0:
            financial_health_risk -= 1.0

        risk_scores["financial_health_risk"] = max(0, min(10, round(financial_health_risk, 1)))

        # 2. Market Risk (based on volatility and beta)
        market_risk = 5.0
        if latest_data:
            volatility = latest_data.get("volatility_30d", 0)
            if volatility:
                if volatility > 40:
                    market_risk = 8.5
                elif volatility > 30:
                    market_risk = 7.0
                elif volatility > 25:
                    market_risk = 6.0
                elif volatility < 15:
                    market_risk = 3.0
                elif volatility < 20:
                    market_risk = 4.0

            # Adjust for beta if available
            beta = latest_data.get("beta", 0)
            if beta > 1.5:
                market_risk += 1.0
            elif beta < 0.8:
                market_risk -= 0.5

        risk_scores["market_risk"] = max(0, min(10, round(market_risk, 1)))

        # 3. Operational Risk (cash flow quality)
        net_income = latest.get("net_income", 0) or 0
        operating_cash_flow = latest.get("operating_cash_flow", 0) or 0

        operational_risk = 5.0
        if net_income > 0 and operating_cash_flow < 0:
            operational_risk = 8.5  # High risk: negative cash despite positive earnings
        elif net_income > 0 and operating_cash_flow > 0:
            cash_quality = operating_cash_flow / net_income if net_income > 0 else 0
            if cash_quality > 1.2:
                operational_risk = 2.5  # Low risk: strong cash generation
            elif cash_quality > 0.9:
                operational_risk = 4.0  # Medium risk
            else:
                operational_risk = 6.0  # Higher risk: weak cash conversion
        elif net_income < 0 and operating_cash_flow > 0:
            operational_risk = 5.5  # Mixed: burning earnings but generating cash

        risk_scores["operational_risk"] = max(0, min(10, round(operational_risk, 1)))

        # 4. Business Model Risk (margin stability)
        if len(quarterly_metrics) >= 4:
            recent_margins = [q.get("net_margin", 0) or 0 for q in quarterly_metrics[-4:]]
            if recent_margins and any(m != 0 for m in recent_margins):
                margin_volatility = np.std(recent_margins)
                avg_margin = np.mean(recent_margins)

                # Check for margin compression
                if len(recent_margins) >= 2:
                    margin_change = recent_margins[-1] - recent_margins[0]
                    if margin_change < -3:  # Declining margins
                        business_model_risk = 7.5
                    elif margin_change < -1:
                        business_model_risk = 6.0
                    elif margin_change > 2:  # Expanding margins
                        business_model_risk = 3.5
                    elif margin_volatility > 5:
                        business_model_risk = 7.0
                    elif margin_volatility > 2:
                        business_model_risk = 5.0
                    else:
                        business_model_risk = 3.5
                else:
                    business_model_risk = 5.0
            else:
                business_model_risk = 6.0  # Missing margin data
        else:
            business_model_risk = 6.0  # Insufficient history

        risk_scores["business_model_risk"] = max(0, min(10, round(business_model_risk, 1)))

        # 5. Growth Risk (from multi-year data)
        growth_risk = 5.0
        if multi_year_trends and multi_year_trends.get("metrics"):
            metrics = multi_year_trends["metrics"]
            revenue_volatility = metrics.get("revenue_volatility", 10)
            revenue_cagr = metrics.get("revenue_cagr", 0)

            if revenue_volatility > 25:
                growth_risk = 7.5
            elif revenue_volatility > 15:
                growth_risk = 6.0
            elif revenue_volatility < 8:
                growth_risk = 3.5

            # Adjust for negative growth
            if revenue_cagr < -5:
                growth_risk += 2.0
            elif revenue_cagr > 15:
                growth_risk -= 1.0

        risk_scores["growth_risk"] = max(0, min(10, round(growth_risk, 1)))

        # Overall risk score (weighted average)
        weights = {
            "financial_health_risk": 0.25,
            "market_risk": 0.20,
            "operational_risk": 0.25,
            "business_model_risk": 0.15,
            "growth_risk": 0.15,
        }

        overall_risk = sum(risk_scores[k] * weights[k] for k in weights.keys())
        risk_scores["overall_risk"] = round(overall_risk, 1)

        # Risk rating
        if overall_risk < 3.0:
            risk_scores["risk_rating"] = "Very Low"
        elif overall_risk < 4.5:
            risk_scores["risk_rating"] = "Low"
        elif overall_risk < 6.0:
            risk_scores["risk_rating"] = "Medium"
        elif overall_risk < 7.5:
            risk_scores["risk_rating"] = "High"
        else:
            risk_scores["risk_rating"] = "Very High"

        return risk_scores

    def _fetch_competitive_positioning_data(self, symbol: str, quarterly_metrics: List[Dict]) -> Dict:
        """
        Fetch competitive positioning data for matrix visualization

        Args:
            symbol: Stock symbol
            quarterly_metrics: Quarterly metrics for the target company

        Returns:
            Dictionary with positioning data for target and peers
        """
        try:
            # Get target company metrics
            if not quarterly_metrics or len(quarterly_metrics) < 4:
                self.main_logger.warning(f"Insufficient quarterly data for positioning matrix")
                return {}

            latest = quarterly_metrics[-1]
            year_ago = quarterly_metrics[-4] if len(quarterly_metrics) >= 4 else quarterly_metrics[0]

            # Calculate target company metrics
            target_revenue = latest.get("revenue", 0) or 0
            year_ago_revenue = year_ago.get("revenue", 0) or 0

            if target_revenue > 0 and year_ago_revenue > 0:
                target_revenue_growth = ((target_revenue / year_ago_revenue) - 1) * 100
            else:
                target_revenue_growth = 0

            target_profit_margin = latest.get("net_margin", 0) or 0

            # Get industry for peer lookup
            industry_query = text(
                """
                SELECT industry, sector
                FROM peer_metrics
                WHERE symbol = :symbol
                LIMIT 1
            """
            )

            with self.db_manager.get_session() as session:
                result = session.execute(industry_query, {"symbol": symbol}).fetchone()

                if not result:
                    self.main_logger.warning(f"No peer group found for {symbol}")
                    return {
                        "target": {
                            "symbol": symbol,
                            "revenue_growth": target_revenue_growth,
                            "profit_margin": target_profit_margin,
                        },
                        "peers": [],
                        "industry": "Unknown",
                        "sector": "Unknown",
                    }

                industry = result[0] if hasattr(result, "__getitem__") else result.industry
                sector = result[1] if hasattr(result, "__getitem__") else result.sector

                # Get peer metrics
                peer_query = text(
                    """
                    SELECT symbol, metrics_data
                    FROM peer_metrics
                    WHERE industry = :industry AND symbol != :symbol
                    LIMIT 15
                """
                )

                peer_result = session.execute(peer_query, {"industry": industry, "symbol": symbol})
                peers = peer_result.fetchall()

            # Build peer positioning data
            peer_positions = []
            for row in peers:
                peer_symbol = row[0] if hasattr(row, "__getitem__") else row.symbol
                metrics = dict(row[1]) if hasattr(row, "__getitem__") else row.metrics_data

                # Extract relevant metrics
                revenue_growth = metrics.get("revenue_growth_yoy", 0)
                profit_margin = metrics.get("profit_margin", 0) or metrics.get("net_margin", 0)

                if revenue_growth is not None and profit_margin is not None:
                    peer_positions.append(
                        {
                            "symbol": peer_symbol,
                            "revenue_growth": float(revenue_growth) if revenue_growth else 0,
                            "profit_margin": float(profit_margin) if profit_margin else 0,
                        }
                    )

            return {
                "target": {
                    "symbol": symbol,
                    "revenue_growth": target_revenue_growth,
                    "profit_margin": target_profit_margin,
                },
                "peers": peer_positions,
                "industry": industry,
                "sector": sector,
            }

        except Exception as e:
            self.main_logger.error(f"Error fetching competitive positioning data: {e}")
            import traceback

            self.main_logger.error(traceback.format_exc())
            return {}

    def _build_peer_performance_leaderboard(self, symbol: str, quarterly_metrics: List[Dict]) -> Dict:
        """
        Build peer performance leaderboard with rankings

        Args:
            symbol: Stock symbol
            quarterly_metrics: Quarterly metrics for the target company

        Returns:
            Dictionary with ranked peer data
        """
        try:
            # Get industry for peer lookup
            industry_query = text(
                """
                SELECT industry, sector
                FROM peer_metrics
                WHERE symbol = :symbol
                LIMIT 1
            """
            )

            with self.db_manager.get_session() as session:
                result = session.execute(industry_query, {"symbol": symbol}).fetchone()

                if not result:
                    self.main_logger.warning(f"No peer group found for leaderboard")
                    return {}

                industry = result[0] if hasattr(result, "__getitem__") else result.industry
                sector = result[1] if hasattr(result, "__getitem__") else result.sector

                # Get all peer metrics including target
                peer_query = text(
                    """
                    SELECT symbol, metrics_data
                    FROM peer_metrics
                    WHERE industry = :industry
                    ORDER BY symbol
                """
                )

                peer_result = session.execute(peer_query, {"industry": industry})
                peers = peer_result.fetchall()

            if not peers:
                return {}

            # Build comprehensive peer performance data
            peer_performance = []
            for row in peers:
                peer_symbol = row[0] if hasattr(row, "__getitem__") else row.symbol
                metrics = dict(row[1]) if hasattr(row, "__getitem__") else row.metrics_data

                # Extract comprehensive metrics
                performance = {
                    "symbol": peer_symbol,
                    "market_cap": metrics.get("market_cap", 0),
                    "revenue": metrics.get("revenue", 0),
                    "revenue_growth": metrics.get("revenue_growth_yoy", 0),
                    "profit_margin": metrics.get("profit_margin", 0) or metrics.get("net_margin", 0),
                    "pe_ratio": metrics.get("pe_ratio", 0),
                    "pb_ratio": metrics.get("pb_ratio", 0),
                    "roe": metrics.get("roe", 0),
                    "debt_to_equity": metrics.get("debt_to_equity", 0),
                    "is_target": peer_symbol == symbol,
                }

                peer_performance.append(performance)

            # Calculate rankings for each metric
            # Revenue growth ranking (higher is better)
            sorted_by_growth = sorted(peer_performance, key=lambda x: x["revenue_growth"] or 0, reverse=True)
            for rank, peer in enumerate(sorted_by_growth, 1):
                for p in peer_performance:
                    if p["symbol"] == peer["symbol"]:
                        p["growth_rank"] = rank

            # Profit margin ranking (higher is better)
            sorted_by_margin = sorted(peer_performance, key=lambda x: x["profit_margin"] or 0, reverse=True)
            for rank, peer in enumerate(sorted_by_margin, 1):
                for p in peer_performance:
                    if p["symbol"] == peer["symbol"]:
                        p["margin_rank"] = rank

            # ROE ranking (higher is better)
            sorted_by_roe = sorted(peer_performance, key=lambda x: x["roe"] or 0, reverse=True)
            for rank, peer in enumerate(sorted_by_roe, 1):
                for p in peer_performance:
                    if p["symbol"] == peer["symbol"]:
                        p["roe_rank"] = rank

            # Market cap ranking (higher is better)
            sorted_by_mcap = sorted(peer_performance, key=lambda x: x["market_cap"] or 0, reverse=True)
            for rank, peer in enumerate(sorted_by_mcap, 1):
                for p in peer_performance:
                    if p["symbol"] == peer["symbol"]:
                        p["mcap_rank"] = rank

            # Calculate composite score (average of ranks, lower is better)
            for peer in peer_performance:
                ranks = [peer.get("growth_rank", 999), peer.get("margin_rank", 999), peer.get("roe_rank", 999)]
                peer["composite_score"] = sum(ranks) / len(ranks)

            # Final ranking by composite score
            peer_performance.sort(key=lambda x: x["composite_score"])
            for rank, peer in enumerate(peer_performance, 1):
                peer["overall_rank"] = rank

            return {
                "peers": peer_performance,
                "industry": industry,
                "sector": sector,
                "total_peers": len(peer_performance),
            }

        except Exception as e:
            self.main_logger.error(f"Error building peer performance leaderboard: {e}")
            import traceback

            self.main_logger.error(traceback.format_exc())
            return {}

    def _calculate_volume_profile(self, symbol: str, latest_data: Dict) -> Dict:
        """
        Calculate volume profile showing volume distribution across price levels

        Args:
            symbol: Stock symbol
            latest_data: Latest technical analysis data with price/volume history

        Returns:
            Dictionary with volume profile data
        """
        try:
            import numpy as np
            import pandas as pd

            # Extract price and volume data from latest_data
            technical_data = latest_data.get("technical", {})
            price_history = technical_data.get("price_history", {})

            if not price_history:
                self.main_logger.warning(f"No price history available for volume profile")
                return {}

            # Convert to DataFrame for easier manipulation
            dates = price_history.get("dates", [])
            closes = price_history.get("close", [])
            volumes = price_history.get("volume", [])
            highs = price_history.get("high", [])
            lows = price_history.get("low", [])

            if not all([dates, closes, volumes]) or len(dates) < 30:
                self.main_logger.warning(f"Insufficient price/volume data for volume profile")
                return {}

            # Use last 90 days for volume profile
            n_days = min(90, len(dates))
            dates = dates[-n_days:]
            closes = closes[-n_days:]
            volumes = volumes[-n_days:]
            highs = highs[-n_days:] if highs else closes
            lows = lows[-n_days:] if lows else closes

            # Create price bins (20 bins from min to max price)
            price_min = min(lows) if lows else min(closes)
            price_max = max(highs) if highs else max(closes)
            price_range = price_max - price_min

            if price_range == 0:
                self.main_logger.warning(f"Invalid price range for volume profile")
                return {}

            num_bins = 20
            bin_size = price_range / num_bins
            price_bins = [price_min + i * bin_size for i in range(num_bins + 1)]

            # Calculate volume at each price level
            volume_at_price = [0] * num_bins

            for i in range(len(dates)):
                close_price = closes[i]
                volume = volumes[i]

                # Find which bin this price falls into
                bin_index = int((close_price - price_min) / bin_size)
                bin_index = max(0, min(num_bins - 1, bin_index))

                volume_at_price[bin_index] += volume

            # Calculate key metrics
            total_volume = sum(volume_at_price)

            # Point of Control (POC): Price level with highest volume
            poc_index = volume_at_price.index(max(volume_at_price))
            poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2
            poc_volume = volume_at_price[poc_index]
            poc_volume_pct = (poc_volume / total_volume * 100) if total_volume > 0 else 0

            # Value Area: Price range containing 70% of volume
            # Sort bins by volume to find the 70% concentration
            bin_volumes_sorted = sorted(enumerate(volume_at_price), key=lambda x: x[1], reverse=True)
            cumulative_volume = 0
            value_area_bins = []
            target_volume = total_volume * 0.70

            for bin_idx, vol in bin_volumes_sorted:
                cumulative_volume += vol
                value_area_bins.append(bin_idx)
                if cumulative_volume >= target_volume:
                    break

            # Get value area high and low
            if value_area_bins:
                value_area_bins.sort()
                value_area_low = price_bins[value_area_bins[0]]
                value_area_high = price_bins[value_area_bins[-1] + 1]
            else:
                value_area_low = price_min
                value_area_high = price_max

            # Current price for context
            current_price = closes[-1]

            # High Volume Nodes (HVN): Bins with > 80th percentile volume
            volume_threshold = np.percentile(volume_at_price, 80)
            hvn_levels = []
            for i, vol in enumerate(volume_at_price):
                if vol >= volume_threshold:
                    price_level = (price_bins[i] + price_bins[i + 1]) / 2
                    hvn_levels.append(
                        {
                            "price": price_level,
                            "volume": vol,
                            "volume_pct": (vol / total_volume * 100) if total_volume > 0 else 0,
                        }
                    )

            # Low Volume Nodes (LVN): Bins with < 20th percentile volume
            lv_threshold = np.percentile(volume_at_price, 20)
            lvn_levels = []
            for i, vol in enumerate(volume_at_price):
                if vol <= lv_threshold and vol > 0:
                    price_level = (price_bins[i] + price_bins[i + 1]) / 2
                    lvn_levels.append(
                        {
                            "price": price_level,
                            "volume": vol,
                            "volume_pct": (vol / total_volume * 100) if total_volume > 0 else 0,
                        }
                    )

            # Build volume profile bins for charting
            profile_bins = []
            for i in range(num_bins):
                profile_bins.append(
                    {
                        "price_low": price_bins[i],
                        "price_high": price_bins[i + 1],
                        "price_mid": (price_bins[i] + price_bins[i + 1]) / 2,
                        "volume": volume_at_price[i],
                        "volume_pct": (volume_at_price[i] / total_volume * 100) if total_volume > 0 else 0,
                    }
                )

            return {
                "profile_bins": profile_bins,
                "current_price": float(current_price),
                "poc_price": float(poc_price),
                "poc_volume_pct": float(poc_volume_pct),
                "value_area_low": float(value_area_low),
                "value_area_high": float(value_area_high),
                "hvn_levels": hvn_levels,
                "lvn_levels": lvn_levels,
                "days_analyzed": n_days,
                "total_volume": int(total_volume),
                "price_range": {"low": float(price_min), "high": float(price_max)},
            }

        except Exception as e:
            self.main_logger.error(f"Error calculating volume profile: {e}")
            import traceback

            self.main_logger.error(traceback.format_exc())
            return {}

    def _fetch_quarterly_metrics(self, symbol: str, limit: int = 12) -> List[Dict]:
        """
        Fetch quarterly financial metrics from database for trend analysis

        Args:
            symbol: Stock symbol
            limit: Number of quarters to fetch (default 12 = 3 years for geometric mean)

        Returns:
            List of quarterly metrics dictionaries
        """
        try:
            import pandas as pd

            query = text(
                """
                SELECT
                    symbol,
                    fiscal_year,
                    fiscal_period,
                    metrics_data
                FROM quarterly_metrics
                WHERE symbol = :symbol
                ORDER BY fiscal_year DESC,
                    CASE fiscal_period
                        WHEN 'Q4' THEN 4
                        WHEN 'Q3' THEN 3
                        WHEN 'Q2' THEN 2
                        WHEN 'Q1' THEN 1
                        ELSE 0
                    END DESC
                LIMIT :limit
            """
            )

            with self.db_manager.get_session() as session:
                result = session.execute(query, {"symbol": symbol, "limit": limit})
                rows = result.fetchall()

            if not rows:
                self.main_logger.warning(f"No quarterly metrics found for {symbol}")
                return []

            # Parse quarterly data
            quarterly_data = []
            for row in rows:
                metrics = dict(row._mapping["metrics_data"]) if hasattr(row, "_mapping") else row.metrics_data

                # Add fiscal period info
                metrics["fiscal_year"] = row.fiscal_year if hasattr(row, "fiscal_year") else row[1]
                metrics["fiscal_period"] = row.fiscal_period if hasattr(row, "fiscal_period") else row[2]
                metrics["period_label"] = f"{metrics['fiscal_year']}-{metrics['fiscal_period']}"

                quarterly_data.append(metrics)

            # Reverse to get chronological order (oldest first)
            quarterly_data.reverse()

            self.main_logger.info(f"Fetched {len(quarterly_data)} quarters of metrics for {symbol}")
            return quarterly_data

        except Exception as e:
            self.main_logger.error(f"Error fetching quarterly metrics for {symbol}: {e}")
            return []

    def _fetch_price_history(self, symbol: str, days: int = 252) -> List[Dict]:
        """
        Fetch historical price data (OHLCV) from database or cache for pattern recognition

        Data source priority:
        1. Primary: stock.tickerdata database table (same source as yahoo_technical.py)
        2. Fallback: Cache (TECHNICAL_DATA from previous analysis)

        Args:
            symbol: Stock symbol
            days: Number of days of history to fetch (default 252 = 1 year)

        Returns:
            List of dictionaries with OHLCV data
        """
        try:
            from datetime import datetime, timedelta

            import pandas as pd

            from investigator.infrastructure.database.market_data import get_market_data_fetcher

            # Try primary source: database (same as yahoo_technical.py)
            try:
                db_fetcher = get_market_data_fetcher(self.config)
                df = db_fetcher.get_stock_data(symbol, days=days)

                if not df.empty and len(df) >= 30:
                    # Convert DataFrame to list of dictionaries
                    price_data = []
                    for date, row in df.iterrows():
                        price_data.append(
                            {
                                "date": date,
                                "open": float(row["Open"]),
                                "high": float(row["High"]),
                                "low": float(row["Low"]),
                                "close": float(row["Close"]),
                                "volume": int(row["Volume"]),
                            }
                        )

                    self.main_logger.info(f"Fetched {len(price_data)} days of price history for {symbol} from database")
                    return price_data
                else:
                    self.main_logger.warning(f"Insufficient data from database for {symbol}, trying cache fallback")

            except Exception as db_error:
                self.main_logger.warning(f"Database fetch failed for {symbol}: {db_error}, trying cache fallback")

            # Fallback: Try cache (TECHNICAL_DATA contains OHLCV + indicators)
            cache_key = (symbol, "technical_data", f"{days}d")
            cached_data = self.cache_manager.get(CacheType.TECHNICAL_DATA, cache_key)

            if cached_data and "dataframe" in cached_data:
                df = cached_data["dataframe"]

                # Select only OHLCV columns needed for pattern recognition
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                if all(col in df.columns for col in required_cols):
                    # Convert DataFrame to list of dictionaries
                    price_data = []
                    for date, row in df.iterrows():
                        price_data.append(
                            {
                                "date": date,
                                "open": float(row["Open"]),
                                "high": float(row["High"]),
                                "low": float(row["Low"]),
                                "close": float(row["Close"]),
                                "volume": int(row["Volume"]),
                            }
                        )

                    self.main_logger.info(f"Fetched {len(price_data)} days of price history for {symbol} from cache")
                    return price_data
                else:
                    self.main_logger.warning(f"Cached data missing OHLCV columns for {symbol}")
                    return []
            else:
                self.main_logger.warning(f"No price history found for {symbol} in database or cache")
                return []

        except Exception as e:
            self.main_logger.error(f"Error fetching price history for {symbol}: {e}")
            return []

    def _calculate_quarterly_trends(self, quarterly_data: List[Dict]) -> Dict:
        """
        Calculate quarter-over-quarter and year-over-year trends

        Args:
            quarterly_data: List of quarterly metrics (chronological order)

        Returns:
            Dictionary with trend analysis
        """
        if len(quarterly_data) < 2:
            return {}

        try:
            trends = {
                "revenue_trend": [],
                "net_income_trend": [],
                "operating_cash_flow_trend": [],
                "margin_trends": [],
                "qoq_growth": {},
                "yoy_growth": {},
            }

            # Calculate Q-o-Q growth for most recent quarter
            if len(quarterly_data) >= 2:
                latest = quarterly_data[-1]
                previous = quarterly_data[-2]

                for metric in ["revenue", "net_income", "operating_cash_flow"]:
                    latest_val = latest.get(metric, 0)
                    prev_val = previous.get(metric, 0)

                    if latest_val and prev_val and prev_val != 0:
                        growth = ((latest_val - prev_val) / abs(prev_val)) * 100
                        trends["qoq_growth"][metric] = round(growth, 2)

            # Calculate Y-o-Y growth (compare same quarter from previous year)
            if len(quarterly_data) >= 5:  # Need at least 5 quarters to compare Q-4 with Q-8
                for i in range(len(quarterly_data) - 4):
                    current = quarterly_data[i + 4]
                    year_ago = quarterly_data[i]

                    for metric in ["revenue", "net_income", "operating_cash_flow"]:
                        current_val = current.get(metric, 0)
                        year_ago_val = year_ago.get(metric, 0)

                        if current_val and year_ago_val and year_ago_val != 0:
                            growth = ((current_val - year_ago_val) / abs(year_ago_val)) * 100
                            if metric not in trends["yoy_growth"]:
                                trends["yoy_growth"][metric] = []
                            trends["yoy_growth"][metric].append(
                                {"period": current["period_label"], "growth": round(growth, 2)}
                            )

            # Extract time series data for charts
            for quarter in quarterly_data:
                period = quarter["period_label"]

                if quarter.get("revenue"):
                    trends["revenue_trend"].append(
                        {"period": period, "value": quarter["revenue"] / 1_000_000}  # Convert to millions
                    )

                if quarter.get("net_income"):
                    trends["net_income_trend"].append({"period": period, "value": quarter["net_income"] / 1_000_000})

                if quarter.get("operating_cash_flow"):
                    trends["operating_cash_flow_trend"].append(
                        {"period": period, "value": quarter["operating_cash_flow"] / 1_000_000}
                    )

                # Calculate margins
                if quarter.get("revenue") and quarter.get("revenue") > 0:
                    net_margin = (quarter.get("net_income", 0) / quarter["revenue"]) * 100
                    op_margin = (quarter.get("operating_income", 0) / quarter["revenue"]) * 100

                    trends["margin_trends"].append(
                        {"period": period, "net_margin": round(net_margin, 2), "operating_margin": round(op_margin, 2)}
                    )

            return trends

        except Exception as e:
            self.main_logger.error(f"Error calculating quarterly trends: {e}")
            return {}

    def _calculate_fundamental_score(self, llm_responses: Dict) -> float:
        """Calculate fundamental score from LLM responses"""
        fundamental_responses = llm_responses.get("fundamental", {})
        if not fundamental_responses:
            return 0.0  # Clear fallback - no data available

        # First try to get from comprehensive analysis
        if "comprehensive" in fundamental_responses:
            comp_resp = fundamental_responses["comprehensive"]
            content = comp_resp.get("content", comp_resp)

            # Handle structured response
            if isinstance(content, dict):
                # Try financial_health_score first, then overall_score
                if "financial_health_score" in content:
                    return float(content["financial_health_score"])
                elif "overall_score" in content:
                    return float(content["overall_score"])

            # Handle string response
            elif isinstance(content, str):
                import re

                # Try to extract from JSON string
                try:
                    import json

                    parsed = json.loads(content)
                    if "financial_health_score" in parsed:
                        return float(parsed["financial_health_score"])
                    elif "overall_score" in parsed:
                        return float(parsed["overall_score"])
                except:
                    # Fall back to regex
                    score_match = re.search(r"(?:Financial Health|Overall|Score)[:\s]*(\d+(?:\.\d+)?)/10", content)
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
                import re

                score_match = re.search(r"(?:Financial Health|Overall|Score)[:\s]*(\d+(?:\.\d+)?)/10", content)
                if score_match:
                    scores.append(float(score_match.group(1)))

        return sum(scores) / len(scores) if scores else 0.0  # Clear fallback - no scores found

    def _calculate_technical_score(self, llm_responses: Dict) -> float:
        """Calculate technical score from structured JSON LLM response"""
        technical_response = llm_responses.get("technical")
        if not technical_response:
            return 0.0  # Clear fallback - no technical data

        content = technical_response.get("content", "")

        # First try to parse as structured JSON (new format)
        if isinstance(content, dict):
            # Check for new structured format
            if "technical_score" in content:
                score_data = content["technical_score"]
                if isinstance(score_data, dict):
                    return float(score_data.get("score", 0.0))
                return float(score_data)
        elif isinstance(content, str):
            # Handle file format with headers - extract JSON part
            json_content = content
            if "=== AI RESPONSE ===" in content:
                json_start = content.find("=== AI RESPONSE ===") + len("=== AI RESPONSE ===")
                json_content = content[json_start:].strip()

            try:
                # Try to parse JSON from string
                parsed = json.loads(json_content)
                if "technical_score" in parsed:
                    score_data = parsed["technical_score"]
                    if isinstance(score_data, dict):
                        return float(score_data.get("score", 0.0))
                    return float(score_data)
            except json.JSONDecodeError:
                pass

            # Fall back to regex for legacy format
            import re

            score_match = re.search(
                r"(?:TECHNICAL[_\s]SCORE|technical_score)[:\s]*(\d+(?:\.\d+)?)", json_content, re.IGNORECASE
            )
            if score_match:
                return float(score_match.group(1))

        return 0.0  # Clear fallback - no score found in response

    def _extract_technical_indicators(self, llm_responses: Dict) -> Dict:
        """Extract technical indicators from structured technical analysis JSON response"""
        technical_response = llm_responses.get("technical")
        if not technical_response:
            return {}

        content = technical_response.get("content", "")
        # Debug: log content type and structure
        self.main_logger.debug(f"Technical response content type: {type(content)}")
        if isinstance(content, str) and len(content) > 0:
            self.main_logger.debug(f"Technical content preview: {content[:100]}...")
        indicators = {}

        # First try to parse as structured JSON
        if isinstance(content, dict):
            # New structured format with comprehensive technical data
            indicators = {
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
                "momentum_signals": self._extract_momentum_signals(content),
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
        elif isinstance(content, str):
            try:
                # Handle file format with headers - extract JSON part
                json_content = content
                if "=== AI RESPONSE ===" in content:
                    json_start = content.find("=== AI RESPONSE ===") + len("=== AI RESPONSE ===")
                    json_content = content[json_start:].strip()

                # Handle responses with <think> prefix - find the JSON part
                json_start = json_content.find("{")
                if json_start >= 0:
                    # Extract JSON part and parse it
                    json_part = json_content[json_start:]
                    # Find the end by counting braces to handle nested JSON
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
                        parsed = json.loads(json_to_parse)
                    else:
                        # Fallback: try to parse the entire json_part
                        parsed = json.loads(json_part)
                else:
                    # No JSON found, try to parse the whole content
                    parsed = json.loads(json_content)
                indicators = {
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
                    "position_sizing": "MODERATE",  # Use default since not in our format
                    "entry_strategy": {},  # Use default since not in our format
                    "volume_analysis": {},  # Use default since not in our format
                    "volatility_analysis": {},  # Use default since not in our format
                    "sector_relative_strength": {},  # Use default since not in our format
                }
            except json.JSONDecodeError:
                # Fall back to legacy format extraction
                indicators = self._extract_legacy_technical_indicators(content)

        # Filter out zero values from support/resistance
        indicators["support_levels"] = [s for s in indicators.get("support_levels", []) if s > 0]
        indicators["resistance_levels"] = [r for r in indicators.get("resistance_levels", []) if r > 0]

        return indicators

    def _extract_momentum_signals(self, content: Dict) -> List[str]:
        """Extract momentum signals from technical analysis response"""
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

    def _extract_legacy_technical_indicators(self, content: str) -> Dict:
        """Extract technical indicators from legacy format response"""
        import re

        indicators = {}

        support_match = re.search(r"support_levels[:\s]*\[([^\]]+)\]", content, re.IGNORECASE)
        resistance_match = re.search(r"resistance_levels[:\s]*\[([^\]]+)\]", content, re.IGNORECASE)
        trend_match = re.search(r'trend_direction[:\s]*["\']?([A-Z]+)["\']?', content, re.IGNORECASE)

        if support_match:
            try:
                indicators["support_levels"] = [float(x.strip()) for x in support_match.group(1).split(",")]
            except:
                indicators["support_levels"] = []

        if resistance_match:
            try:
                indicators["resistance_levels"] = [float(x.strip()) for x in resistance_match.group(1).split(",")]
            except:
                indicators["resistance_levels"] = []

        if trend_match:
            indicators["trend_direction"] = trend_match.group(1).upper()

        return indicators

    def _extract_sec_comprehensive_data(self, llm_responses: Dict) -> Dict:
        """Extract all valuable data from SEC comprehensive analysis"""
        fundamental_responses = llm_responses.get("fundamental", {})
        if "comprehensive" not in fundamental_responses:
            return {}

        comp_resp = fundamental_responses["comprehensive"]
        content = comp_resp.get("content", comp_resp)

        # Handle structured response
        if isinstance(content, dict):
            return {
                "financial_health_score": content.get("financial_health_score", 0.0),
                "business_quality_score": content.get("business_quality_score", 0.0),
                "growth_prospects_score": content.get("growth_prospects_score", 0.0),
                "data_quality_score": (
                    content.get("data_quality_score", {}).get("score", 0.0)
                    if isinstance(content.get("data_quality_score"), dict)
                    else content.get("data_quality_score", 0.0)
                ),
                "overall_score": content.get("overall_score", 0.0),
                "investment_thesis": content.get("investment_thesis", ""),
                "key_insights": content.get("key_insights", []),
                "key_risks": content.get("key_risks", []),
                "trend_analysis": content.get("trend_analysis", {}),
                "confidence_level": content.get("confidence_level", "MEDIUM"),
            }

        # Handle string response (legacy format)
        elif isinstance(content, str):
            try:
                parsed = json.loads(content)
                return {
                    "financial_health_score": parsed.get("financial_health_score", 0.0),
                    "business_quality_score": parsed.get("business_quality_score", 0.0),
                    "growth_prospects_score": parsed.get("growth_prospects_score", 0.0),
                    "data_quality_score": (
                        parsed.get("data_quality_score", {}).get("score", 0.0)
                        if isinstance(parsed.get("data_quality_score"), dict)
                        else parsed.get("data_quality_score", 0.0)
                    ),
                    "overall_score": parsed.get("overall_score", 0.0),
                    "investment_thesis": parsed.get("investment_thesis", ""),
                    "key_insights": parsed.get("key_insights", []),
                    "key_risks": parsed.get("key_risks", []),
                    "trend_analysis": parsed.get("trend_analysis", {}),
                    "confidence_level": parsed.get("confidence_level", "MEDIUM"),
                }
            except:
                return {}

        return {}

    def _create_recommendation_from_llm_data(
        self, symbol: str, sec_data: Dict, tech_indicators: Dict, current_price: float, overall_score: float
    ) -> Dict:
        """Create investment recommendation by combining SEC comprehensive and technical analysis data"""

        # Extract key scores and data (use fundamental_score instead of financial_health_score)
        business_quality = sec_data.get("business_quality_score", 0.0)
        fundamental_score = sec_data.get("financial_health_score", 0.0)  # This becomes our fundamental score
        growth_score = sec_data.get("growth_prospects_score", 0.0)  # Use growth_score not growth_prospects
        data_quality = sec_data.get("data_quality_score", 0.0)
        sec_confidence = sec_data.get("confidence_level", "MEDIUM")

        # Technical data
        tech_trend = tech_indicators.get("trend_direction", "NEUTRAL")
        tech_recommendation = tech_indicators.get("recommendation", "HOLD")
        support_levels = tech_indicators.get("support_levels", [])
        resistance_levels = tech_indicators.get("resistance_levels", [])
        tech_risks = tech_indicators.get("risk_factors", [])

        # Combine recommendations - prioritize fundamental for long-term view
        if fundamental_score >= 8.0 and business_quality >= 8.0:
            if tech_trend in ["BULLISH", "NEUTRAL"]:
                final_recommendation = "BUY"
                confidence = "HIGH" if tech_trend == "BULLISH" else "MEDIUM"
            else:  # BEARISH
                final_recommendation = "HOLD"  # Strong fundamentals but poor technicals
                confidence = "MEDIUM"
        elif fundamental_score >= 6.0 and business_quality >= 6.0:
            if tech_trend == "BULLISH":
                final_recommendation = "BUY"
                confidence = "MEDIUM"
            elif tech_trend == "BEARISH":
                final_recommendation = "HOLD"
                confidence = "LOW"
            else:
                final_recommendation = "HOLD"
                confidence = "MEDIUM"
        else:  # Weak fundamentals
            if tech_trend == "BEARISH":
                final_recommendation = "SELL"
                confidence = "MEDIUM"
            else:
                final_recommendation = "HOLD"
                confidence = "LOW"

        # Adjust confidence based on data quality
        if data_quality < 5.0:
            confidence = "LOW"
        elif data_quality >= 8.0 and confidence == "MEDIUM":
            confidence = "HIGH"

        # Create combined investment thesis
        sec_thesis = sec_data.get("investment_thesis", "")
        if sec_thesis and tech_indicators:
            investment_thesis = f"{sec_thesis} Technical analysis shows {tech_trend.lower()} trend with {tech_recommendation.lower()} recommendation."
        elif sec_thesis:
            investment_thesis = sec_thesis
        else:
            investment_thesis = f"Based on fundamental score of {fundamental_score:.1f} and business quality of {business_quality:.1f}, with {tech_trend.lower()} technical trend."

        # Combine key insights and risks
        sec_insights = sec_data.get("key_insights", [])
        sec_risks = sec_data.get("key_risks", [])

        # Add technical insights
        tech_insights = []
        if support_levels:
            tech_insights.append(f"Key support levels at ${', $'.join([f'{s:.2f}' for s in support_levels[:3]])}")
        if resistance_levels:
            tech_insights.append(f"Key resistance levels at ${', $'.join([f'{r:.2f}' for r in resistance_levels[:3]])}")

        all_insights = sec_insights + tech_insights
        all_risks = sec_risks + tech_risks

        # Calculate position sizing based on combined analysis
        if final_recommendation == "BUY":
            if confidence == "HIGH" and business_quality >= 9.0:
                position_size = "LARGE"
            elif confidence in ["HIGH", "MEDIUM"]:
                position_size = "MODERATE"
            else:
                position_size = "SMALL"
        elif final_recommendation == "SELL":
            position_size = "AVOID"
        else:  # HOLD
            position_size = "SMALL"

        # Time horizon based on fundamental strength
        if business_quality >= 8.0 and fundamental_score >= 8.0:
            time_horizon = "LONG-TERM"
        elif business_quality >= 6.0:
            time_horizon = "MEDIUM-TERM"
        else:
            time_horizon = "SHORT-TERM"

        # Calculate price targets using support/resistance
        price_target = None
        stop_loss = None
        if resistance_levels and final_recommendation == "BUY":
            price_target = max(resistance_levels)
        if support_levels and final_recommendation in ["BUY", "HOLD"]:
            stop_loss = min(support_levels) * 0.95  # 5% below support

        return {
            "overall_score": overall_score,
            "fundamental_score": fundamental_score,
            "technical_score": tech_indicators.get("technical_score", 0.0),
            "business_quality_score": business_quality,
            "growth_score": growth_score,
            "data_quality_score": data_quality,
            "investment_recommendation": {"recommendation": final_recommendation, "confidence": confidence},
            "investment_thesis": investment_thesis,
            "position_size": position_size,
            "time_horizon": time_horizon,
            "price_target": price_target,
            "stop_loss": stop_loss,
            "key_catalysts": all_insights[:5],  # Top 5 insights as catalysts
            "downside_risks": all_risks[:5],  # Top 5 risks
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "trend_direction": tech_trend,
            "momentum_signals": tech_indicators.get("momentum_signals", []),
            "confidence_level": confidence,
            "source": "direct_llm_extraction",
        }

    def _calculate_weighted_score(self, fundamental_score: float, technical_score: float) -> float:
        """Calculate weighted overall score"""
        if fundamental_score is None or technical_score is None:
            return 5.0

        fund_weight = self.config.analysis.fundamental_weight
        tech_weight = self.config.analysis.technical_weight

        # Adjust weights for extreme scores
        if fundamental_score >= 8.5 or fundamental_score <= 2.5:
            fund_weight *= 1.2

        if technical_score >= 8.5 or technical_score <= 2.5:
            tech_weight *= 1.1

        total_weight = fund_weight + tech_weight

        if total_weight == 0:
            return 0.0  # Clear fallback - no weights

        norm_fund_weight = fund_weight / total_weight
        norm_tech_weight = tech_weight / total_weight

        overall_score = fundamental_score * norm_fund_weight + technical_score * norm_tech_weight

        return round(overall_score, 1)

    def _calculate_data_quality_detailed(
        self, symbol: str, llm_responses: Dict, quarterly_metrics: List, latest_data: Dict
    ) -> Dict:
        """
        Calculate detailed data quality score with component breakdown

        Args:
            symbol: Stock symbol
            llm_responses: Dictionary of LLM responses
            quarterly_metrics: List of quarterly metrics
            latest_data: Latest market data

        Returns:
            Dictionary with overall score, grade, and component scores
        """
        scores = []
        details = {}

        # Component 1: LLM Response Completeness
        expected_llm_types = ["fundamental", "technical", "quarterly_summary"]
        available_llm = sum(1 for t in expected_llm_types if t in llm_responses)
        llm_completeness = (available_llm / len(expected_llm_types)) * 100
        scores.append(llm_completeness)
        details["llm_completeness"] = llm_completeness

        # Component 2: Quarterly Data Availability
        if quarterly_metrics:
            quarters_available = len(quarterly_metrics)
            quarterly_completeness = min((quarters_available / 8) * 100, 100)
        else:
            quarterly_completeness = 0
        scores.append(quarterly_completeness)
        details["quarterly_completeness"] = quarterly_completeness

        # Component 3: Market Data Freshness
        if latest_data:
            # Assume fresh if we have it
            market_freshness = 100
        else:
            market_freshness = 0
        scores.append(market_freshness)
        details["market_freshness"] = market_freshness

        # Component 4: Peer Data Availability
        try:
            query = text(
                """
                SELECT COUNT(*) FROM peer_metrics
                WHERE industry = (SELECT industry FROM peer_metrics WHERE symbol = :symbol LIMIT 1)
            """
            )
            with self.db_manager.get_session() as session:
                peer_count = session.execute(query, {"symbol": symbol}).scalar()
            peer_availability = min((peer_count / 10) * 100, 100) if peer_count else 0
        except Exception as e:
            self.main_logger.warning(f"Could not get peer data availability: {e}")
            peer_availability = 0
        scores.append(peer_availability)
        details["peer_availability"] = peer_availability

        # Overall score
        overall_score = sum(scores) / len(scores)

        # Letter grade
        if overall_score >= 90:
            grade = "A"
        elif overall_score >= 80:
            grade = "B"
        elif overall_score >= 70:
            grade = "C"
        elif overall_score >= 60:
            grade = "D"
        else:
            grade = "F"

        return {
            "overall_score": round(overall_score, 1),
            "grade": grade,
            "components": details,
            "timestamp": datetime.now(timezone.utc),
        }

    def _assess_data_quality(self, llm_responses: Dict, latest_data: Dict) -> float:
        """Assess overall data quality and completeness, prioritizing SEC comprehensive analysis

        Returns:
            float: Data quality score on 1-10 scale
        """
        # First, try to get data quality score from SEC comprehensive analysis
        comprehensive_analysis = llm_responses.get("fundamental", {}).get("comprehensive", {})
        if isinstance(comprehensive_analysis, dict):
            # Direct score from comprehensive analysis
            if "data_quality_score" in comprehensive_analysis:
                score_data = comprehensive_analysis["data_quality_score"]
                if isinstance(score_data, dict):
                    return float(score_data.get("score", 0.0))  # Already in 1-10 scale
                return float(score_data)  # Already in 1-10 scale

            # Extract from response content if it's nested
            content = comprehensive_analysis.get("content", {})
            if isinstance(content, dict) and "data_quality_score" in content:
                score_data = content["data_quality_score"]
                if isinstance(score_data, dict):
                    return float(score_data.get("score", 0.0))  # Already in 1-10 scale
                return float(score_data)  # Already in 1-10 scale

        # Fallback to traditional data quality assessment (convert to 1-10 scale)
        quality_score = 0.0

        # Check fundamental data availability (max 4 points)
        if llm_responses.get("fundamental"):
            quality_score += 4.0
            if len(llm_responses["fundamental"]) >= 3:  # Multiple quarters
                quality_score += 1.0

        # Check technical data availability (max 3 points)
        if llm_responses.get("technical"):
            quality_score += 3.0

        # Check data freshness (max 2 points)
        if latest_data.get("technical", {}).get("current_price"):
            quality_score += 1.0

        if latest_data.get("fundamental"):
            quality_score += 1.0

        return min(quality_score, 10.0)  # Cap at 10.0

    def _parse_synthesis_response(self, response: str) -> Dict:
        """Parse the synthesis LLM response"""
        import re

        result = {
            "recommendation": "HOLD",
            "confidence": "MEDIUM",
            "investment_thesis": "",
            "key_catalysts": [],
            "key_risks": [],
            "price_targets": {},
            "position_size": "MODERATE",
            "time_horizon": "MEDIUM-TERM",
            "entry_strategy": "",
            "exit_strategy": "",
        }

        try:
            # Extract final recommendation
            rec_match = re.search(r"FINAL RECOMMENDATION[:\s]*\*?\*?\s*\[?([A-Z\s]+)\]?", response, re.IGNORECASE)
            if rec_match:
                rec_text = rec_match.group(1).strip().upper()
                if "STRONG BUY" in rec_text:
                    result["recommendation"] = "STRONG BUY"
                elif "STRONG SELL" in rec_text:
                    result["recommendation"] = "STRONG SELL"
                elif "BUY" in rec_text:
                    result["recommendation"] = "BUY"
                elif "SELL" in rec_text:
                    result["recommendation"] = "SELL"
                else:
                    result["recommendation"] = "HOLD"

            # Extract confidence level
            conf_match = re.search(r"CONFIDENCE LEVEL[:\s]*\*?\*?\s*\[?([A-Z]+)\]?", response, re.IGNORECASE)
            if conf_match:
                result["confidence"] = conf_match.group(1).strip().upper()

            # Extract investment thesis
            thesis_match = re.search(
                r"INVESTMENT THESIS[:\s]*\*?\*?(.*?)(?=\*\*[A-Z]|\n\n)", response, re.IGNORECASE | re.DOTALL
            )
            if thesis_match:
                result["investment_thesis"] = thesis_match.group(1).strip()

            # Extract catalysts
            catalysts_match = re.search(
                r"KEY CATALYSTS[:\s]*\*?\*?(.*?)(?=\*\*[A-Z]|\n\n)", response, re.IGNORECASE | re.DOTALL
            )
            if catalysts_match:
                catalysts_text = catalysts_match.group(1)
                result["key_catalysts"] = [cat.strip() for cat in re.findall(r"[•\-]\s*(.+)", catalysts_text)]

            # Extract risks
            risks_match = re.search(
                r"RISK ASSESSMENT[:\s]*\*?\*?(.*?)(?=\*\*[A-Z]|\n\n)", response, re.IGNORECASE | re.DOTALL
            )
            if risks_match:
                risks_text = risks_match.group(1)
                result["key_risks"] = [risk.strip() for risk in re.findall(r"[•\-]\s*(.+)", risks_text)]

            # Extract price targets
            target_match = re.search(r"12-month.*?Target[:\s]*\$?([\d.]+)", response, re.IGNORECASE)
            if target_match:
                result["price_targets"]["12_month"] = float(target_match.group(1))

            # Extract position size
            pos_match = re.search(r"POSITION SIZING[:\s]*\*?\*?\s*\[?([A-Z\s\/%]+)\]?", response, re.IGNORECASE)
            if pos_match:
                pos_text = pos_match.group(1).strip().upper()
                if "LARGE" in pos_text or "CONCENTRATED" in pos_text:
                    result["position_size"] = "LARGE"
                elif "SMALL" in pos_text or "STARTER" in pos_text:
                    result["position_size"] = "SMALL"
                else:
                    result["position_size"] = "MODERATE"

            # Extract time horizon
            horizon_match = re.search(r"TIME HORIZON[:\s]*\*?\*?\s*\[?([A-Z\s\-]+)\]?", response, re.IGNORECASE)
            if horizon_match:
                result["time_horizon"] = horizon_match.group(1).strip().upper()

        except Exception as e:
            self.main_logger.warning(f"Error parsing synthesis response: {e}")

        return result

    def _extract_income_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """Extract income statement score from responses"""
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

    def _extract_cashflow_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """Extract cash flow score from responses"""
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

    def _extract_balance_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """Extract balance sheet score from responses"""
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

    def _extract_growth_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """Extract growth prospects score from responses"""
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

    def _extract_value_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """Extract value investment score from responses"""
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

    def _extract_business_quality_score(self, llm_responses: Dict, ai_recommendation: Dict) -> float:
        """
        Extract business quality score from SEC comprehensive analysis.

        This method works backwards from the comprehensive SEC analysis which aggregates
        quarterly data to assess business quality based on:
        - Core business concepts and tags identified across quarters
        - Revenue quality and consistency patterns
        - Operational efficiency metrics
        - Competitive positioning indicators
        - Management effectiveness signals
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
                import json

                parsed = json.loads(comprehensive_analysis)
                if "business_quality_score" in parsed:
                    return float(parsed["business_quality_score"])
            except:
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
            quality_score = self._analyze_quarterly_business_quality(content, period_key)
            if quality_score > 0:
                quality_indicators.append(quality_score)

        # Calculate average business quality from quarterly analyses
        if quality_indicators:
            avg_quality = sum(quality_indicators) / len(quality_indicators)

            # Apply weighting based on data consistency and trends
            consistency_bonus = self._calculate_consistency_bonus(quality_indicators)
            final_score = min(10.0, max(1.0, avg_quality + consistency_bonus))

            return final_score

        # Ultimate fallback: Return 0 to indicate no business quality score available
        return 0.0

    def _analyze_quarterly_business_quality(self, content: str, period: str) -> float:
        """Analyze individual quarterly content for business quality indicators"""
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

    def _calculate_consistency_bonus(self, quality_indicators: List[float]) -> float:
        """Calculate bonus for consistent business quality across quarters"""
        if len(quality_indicators) < 2:
            return 0.0

        # Calculate standard deviation
        mean_quality = sum(quality_indicators) / len(quality_indicators)
        variance = sum((x - mean_quality) ** 2 for x in quality_indicators) / len(quality_indicators)
        std_dev = variance**0.5

        # Lower standard deviation = more consistent = higher bonus
        # Scale: 0-1 point bonus based on consistency
        max_bonus = 1.0
        consistency_bonus = max(0.0, max_bonus - (std_dev / 2.0))

        return consistency_bonus

    def _determine_final_recommendation(
        self, overall_score: float, ai_recommendation: Dict, data_quality: float
    ) -> Dict:
        """Determine final recommendation with risk management"""
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

    def _calculate_price_target(
        self, symbol: str, llm_responses: Dict, ai_recommendation: Dict, current_price: float
    ) -> float:
        """Calculate sophisticated price target"""
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
            # Log warning and use a placeholder
            self.main_logger.warning(
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
        self.main_logger.info(
            f"Calculated price target for {symbol}: ${price_target:.2f} (current: ${current_price:.2f}, score: {overall_score:.1f})"
        )

        return price_target

    def _calculate_stop_loss(self, current_price: float, recommendation: Dict, overall_score: float) -> float:
        """Calculate stop loss based on risk management"""
        if not current_price or current_price <= 0:
            return 0

        # Base stop loss percentage on score and recommendation
        rec_type = recommendation.get("recommendation", "HOLD")

        if "STRONG BUY" in rec_type:
            stop_loss_pct = 0.12  # 12% stop loss
        elif "BUY" in rec_type:
            stop_loss_pct = 0.10  # 10% stop loss
        elif "HOLD" in rec_type:
            stop_loss_pct = 0.08  # 8% stop loss
        else:  # SELL
            stop_loss_pct = 0.05  # 5% stop loss

        # Adjust for overall score
        if overall_score < 4.0:
            stop_loss_pct *= 0.5  # Tighter stop for low conviction

        return round(current_price * (1 - stop_loss_pct), 2)

    def _extract_position_size(self, ai_recommendation: Dict) -> str:
        """Extract position size recommendation"""
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

    def _extract_catalysts(self, ai_recommendation: Dict) -> List[str]:
        """Extract key catalysts from recommendation"""
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

    def _extract_insights_from_text(self, text_details: str) -> tuple[List[str], List[str]]:
        """Extract insights and risks from additional text details beyond JSON"""
        import re

        insights = []
        risks = []

        if not text_details:
            return insights, risks

        # Clean and normalize text
        text = text_details.strip()

        # Extract insights using various patterns
        insight_patterns = [
            r"(?:key\s+)?insights?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)",
            r"(?:important\s+)?findings?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)",
            r"(?:notable\s+)?observations?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)",
            r"(?:investment\s+)?highlights?[:\s]+(.*?)(?=\n\n|\nkey\s+risks?|\n[A-Z]|$)",
        ]

        for pattern in insight_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Extract bullet points or numbered items
                bullet_items = re.findall(r"[•\-\*]\s*(.+)", match)
                numbered_items = re.findall(r"\d+\.\s*(.+)", match)

                # Add bullet items
                for item in bullet_items:
                    clean_item = item.strip()
                    if len(clean_item) > 10 and clean_item not in insights:
                        insights.append(clean_item[:200])  # Limit length

                # Add numbered items
                for item in numbered_items:
                    clean_item = item.strip()
                    if len(clean_item) > 10 and clean_item not in insights:
                        insights.append(clean_item[:200])  # Limit length

        # Extract risks using various patterns
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
                # Extract bullet points or numbered items
                bullet_items = re.findall(r"[•\-\*]\s*(.+)", match)
                numbered_items = re.findall(r"\d+\.\s*(.+)", match)

                # Add bullet items
                for item in bullet_items:
                    clean_item = item.strip()
                    if len(clean_item) > 10 and clean_item not in risks:
                        risks.append(clean_item[:200])  # Limit length

                # Add numbered items
                for item in numbered_items:
                    clean_item = item.strip()
                    if len(clean_item) > 10 and clean_item not in risks:
                        risks.append(clean_item[:200])  # Limit length

        # If no structured patterns found, try to extract from general text
        if not insights and not risks:
            # Split text into sentences and look for insight/risk indicators
            sentences = re.split(r"[.!?]+", text)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue

                # Check for insight indicators
                insight_indicators = ["strength", "opportunity", "advantage", "positive", "growth", "improve"]
                if any(indicator in sentence.lower() for indicator in insight_indicators):
                    if len(insights) < 3:  # Limit to avoid noise
                        insights.append(sentence[:200])

                # Check for risk indicators
                risk_indicators = ["risk", "concern", "challenge", "threat", "weakness", "decline", "pressure"]
                if any(indicator in sentence.lower() for indicator in risk_indicators):
                    if len(risks) < 3:  # Limit to avoid noise
                        risks.append(sentence[:200])

        # Limit total items
        insights = insights[:5]  # Max 5 insights
        risks = risks[:5]  # Max 5 risks

        return insights, risks

    def _extract_comprehensive_risks(
        self, llm_responses: Dict, ai_recommendation: Dict, additional_risks: List[str] = None
    ) -> List[str]:
        """Extract and prioritize comprehensive risk factors"""
        import re

        risks = []

        # From AI synthesis
        if isinstance(ai_recommendation, dict):
            ai_risks = ai_recommendation.get("key_risks", [])
            if isinstance(ai_risks, list):
                risks.extend(ai_risks[:3])

        # Add additional risks from text details if available
        if additional_risks:
            risks.extend(additional_risks)

        # Extract from fundamental responses
        for resp in llm_responses.get("fundamental", {}).values():
            content = resp.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            risk_section = re.search(r"risk[s]?[:\s]*(.*?)(?=\n\n|\d+\.)", content, re.IGNORECASE | re.DOTALL)
            if risk_section:
                risk_items = re.findall(r"[•\-]\s*(.+)", risk_section.group(1))
                risks.extend(risk_items[:2])

        # Deduplicate and limit
        unique_risks = []
        seen = set()
        for risk in risks:
            risk_lower = risk.lower().strip()
            if risk_lower not in seen and len(risk_lower) > 10:
                seen.add(risk_lower)
                unique_risks.append(risk)

        return unique_risks[:8] if unique_risks else ["Limited risk data available"]

    def _extract_comprehensive_insights(
        self, llm_responses: Dict, ai_recommendation: Dict, additional_insights: List[str] = None
    ) -> List[str]:
        """Extract and prioritize comprehensive insights"""
        import re

        insights = []

        # Add additional insights from text details if available
        if additional_insights:
            insights.extend(additional_insights)

        # From AI synthesis catalysts
        if isinstance(ai_recommendation, dict):
            catalysts = ai_recommendation.get("key_catalysts", [])
            if isinstance(catalysts, list):
                insights.extend([f"Catalyst: {cat}" for cat in catalysts[:2]])

        # Extract key findings from responses
        for resp in llm_responses.get("fundamental", {}).values():
            content = resp.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            insights_section = re.search(
                r"key\s+(?:insight|finding)[s]?[:\s]*(.*?)(?=\n\n|\d+\.)", content, re.IGNORECASE | re.DOTALL
            )
            if insights_section:
                insight_items = re.findall(r"[•\-]\s*(.+)", insights_section.group(1))
                insights.extend(insight_items[:2])

        # Technical insights
        tech_resp = llm_responses.get("technical")
        if tech_resp:
            content = tech_resp.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            tech_insights = re.findall(
                r"KEY INSIGHTS[:\s]*\*?\*?(.*?)(?=\*\*[A-Z]|\n\n)", content, re.IGNORECASE | re.DOTALL
            )
            if tech_insights:
                tech_items = re.findall(r"[•\-]\s*(.+)", tech_insights[0])
                insights.extend([f"Technical: {item}" for item in tech_items[:2]])

        # Deduplicate and limit
        unique_insights = []
        seen = set()
        for insight in insights:
            insight_lower = insight.lower().strip()
            if insight_lower not in seen and len(insight_lower) > 10:
                seen.add(insight_lower)
                unique_insights.append(insight)

        return unique_insights[:8] if unique_insights else ["Analysis insights pending"]

    def _save_synthesis_llm_response(
        self, symbol: str, prompt: str, response: str, processing_time_ms: int, synthesis_mode: str = "comprehensive"
    ):
        """Save synthesis LLM response to database and disk"""
        try:
            # Prepare data for DAO
            response_obj = {"type": "text", "content": response}

            metadata = {
                "processing_time_ms": processing_time_ms,
                "response_length": len(response),
                "timestamp": datetime.now().isoformat(),
                "synthesis_type": "full",
                "model": self.config.ollama.models.get("synthesis", "deepseek-r1:32b"),
            }

            # Store full prompt directly
            prompt_data = prompt

            model_info = {
                "model": metadata["model"],
                "temperature": self.config.ollama.temperatures.get_temperature(
                    "balanced"
                ),  # Synthesis uses balanced temperature (0.1)
                "top_p": 0.9,
                "num_ctx": 32768,
                "num_predict": 4096,
            }

            # Determine fiscal period from available data
            fiscal_year, fiscal_period = self._get_latest_fiscal_period()

            # Save to cache using cache manager with synthesis mode-specific llm_type
            # Use intelligent defaults: SYNTHESIS as form_type for synthesis analysis
            llm_type = f"synthesis_{synthesis_mode}"  # synthesis_comprehensive or synthesis_quarterly
            cache_key = {
                "symbol": symbol,
                "form_type": "SYNTHESIS",  # Intelligent default for synthesis analysis
                "period": f"{fiscal_year}-{fiscal_period}",
                "fiscal_year": fiscal_year,  # Separate key for file pattern
                "fiscal_period": fiscal_period,  # Separate key for file pattern
                "llm_type": llm_type,
            }
            cache_value = {
                "prompt": prompt_data,
                "model_info": model_info,
                "response": response_obj,
                "metadata": metadata,
            }

            success = self.cache_manager.set(CacheType.LLM_RESPONSE, cache_key, cache_value)

            if success:
                self.main_logger.info(f"💾 Stored synthesis LLM response for {symbol}")
            else:
                self.main_logger.error(f"Failed to store synthesis LLM response for {symbol}")

            # Also save the prompt and response as separate text files for visibility
            symbol_cache_dir = self.llm_cache_dir / symbol
            symbol_cache_dir.mkdir(parents=True, exist_ok=True)

            # Use mode-specific filenames to avoid overlap
            mode_suffix = "_comprehensive" if synthesis_mode == "comprehensive" else "_quarterly"

            # Save prompt
            prompt_file = symbol_cache_dir / f"prompt_synthesis{mode_suffix}.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)

            # Save response
            response_file = symbol_cache_dir / f"response_synthesis{mode_suffix}.txt"
            with open(response_file, "w", encoding="utf-8") as f:
                f.write(response)

            self.main_logger.info(
                f"💾 Saved synthesis prompt and response to {symbol_cache_dir} (mode: {synthesis_mode})"
            )

        except Exception as e:
            self.main_logger.error(f"Error saving synthesis LLM response: {e}")

    def _get_previous_recommendation(self, symbol: str) -> Optional[Dict]:
        """
        Get the most recent previous recommendation for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with previous recommendation data, or None if not found
        """
        try:
            query = text(
                """
                SELECT overall_score, fundamental_score, technical_score,
                       recommendation, confidence, price_target, current_price,
                       analysis_timestamp
                FROM synthesis_results
                WHERE symbol = :symbol
                ORDER BY analysis_timestamp DESC
                LIMIT 1
            """
            )

            with self.db_manager.get_session() as session:
                result = session.execute(query, {"symbol": symbol}).fetchone()

                if result:
                    return {
                        "symbol": symbol,
                        "overall_score": float(result[0]) if result[0] else 0.0,
                        "fundamental_score": float(result[1]) if result[1] else 0.0,
                        "technical_score": float(result[2]) if result[2] else 0.0,
                        "recommendation": result[3],
                        "confidence": result[4],
                        "price_target": float(result[5]) if result[5] else None,
                        "current_price": float(result[6]) if result[6] else None,
                        "analysis_timestamp": result[7],
                    }

                return None

        except Exception as e:
            self.main_logger.error(f"Error retrieving previous recommendation for {symbol}: {e}")
            return None

    def _save_synthesis_results(self, symbol: str, recommendation: InvestmentRecommendation):
        """Save synthesis results to database"""
        try:
            import json

            # Prepare data for insertion
            insert_query = text(
                """
                INSERT INTO synthesis_results (
                    symbol, analysis_timestamp, overall_score, fundamental_score, technical_score,
                    income_score, cashflow_score, balance_score, growth_score, value_score,
                    business_quality_score, recommendation, confidence, price_target, current_price,
                    investment_thesis, time_horizon, position_size, key_catalysts, key_risks,
                    key_insights, entry_strategy, exit_strategy, stop_loss, data_quality_score
                ) VALUES (
                    :symbol, :analysis_timestamp, :overall_score, :fundamental_score, :technical_score,
                    :income_score, :cashflow_score, :balance_score, :growth_score, :value_score,
                    :business_quality_score, :recommendation, :confidence, :price_target, :current_price,
                    :investment_thesis, :time_horizon, :position_size, :key_catalysts, :key_risks,
                    :key_insights, :entry_strategy, :exit_strategy, :stop_loss, :data_quality_score
                )
            """
            )

            with self.db_manager.get_session() as session:
                session.execute(
                    insert_query,
                    {
                        "symbol": symbol,
                        "analysis_timestamp": recommendation.analysis_timestamp,
                        "overall_score": recommendation.overall_score,
                        "fundamental_score": recommendation.fundamental_score,
                        "technical_score": recommendation.technical_score,
                        "income_score": recommendation.income_score,
                        "cashflow_score": recommendation.cashflow_score,
                        "balance_score": recommendation.balance_score,
                        "growth_score": recommendation.growth_score,
                        "value_score": recommendation.value_score,
                        "business_quality_score": recommendation.business_quality_score,
                        "recommendation": recommendation.recommendation,
                        "confidence": recommendation.confidence,
                        "price_target": recommendation.price_target,
                        "current_price": recommendation.current_price,
                        "investment_thesis": (
                            recommendation.investment_thesis[:5000] if recommendation.investment_thesis else None
                        ),  # Truncate if too long
                        "time_horizon": recommendation.time_horizon,
                        "position_size": recommendation.position_size,
                        "key_catalysts": (
                            json.dumps(recommendation.key_catalysts) if recommendation.key_catalysts else None
                        ),
                        "key_risks": json.dumps(recommendation.key_risks) if recommendation.key_risks else None,
                        "key_insights": (
                            json.dumps(recommendation.key_insights) if recommendation.key_insights else None
                        ),
                        "entry_strategy": recommendation.entry_strategy,
                        "exit_strategy": recommendation.exit_strategy,
                        "stop_loss": recommendation.stop_loss,
                        "data_quality_score": recommendation.data_quality_score,
                    },
                )
                session.commit()

            self.main_logger.info(
                f"✅ Saved synthesis results to database for {symbol}: {recommendation.recommendation} (Score: {recommendation.overall_score})"
            )

        except Exception as e:
            self.main_logger.error(f"Failed to save synthesis results for {symbol}: {e}")
            # Don't fail the entire synthesis if database save fails
            pass

    def _get_market_summary(self) -> Dict:
        """Get market summary data for weekly reports"""
        # Placeholder - would fetch real market data
        return {
            "sp500": "4,500.00",
            "sp500_week_change": "+1.2%",
            "sp500_ytd_change": "+15.3%",
            "nasdaq": "14,000.00",
            "nasdaq_week_change": "+2.1%",
            "nasdaq_ytd_change": "+22.5%",
            "dow": "35,000.00",
            "dow_week_change": "+0.8%",
            "dow_ytd_change": "+8.2%",
            "commentary": "Markets showed resilience this week despite mixed economic data.",
        }

    def _calculate_portfolio_performance(self, recommendations: List[Dict]) -> Dict:
        """Calculate portfolio performance metrics"""
        if not recommendations:
            return {}

        # Calculate aggregate metrics
        avg_score = sum(r["overall_score"] for r in recommendations) / len(recommendations)
        buy_count = sum(1 for r in recommendations if "BUY" in r["recommendation"])
        win_rate = buy_count / len(recommendations) * 100 if recommendations else 0

        # Find best/worst performers (placeholder logic)
        sorted_by_score = sorted(recommendations, key=lambda x: x["overall_score"], reverse=True)

        return {
            "week_return": "+2.3%",  # Placeholder
            "month_return": "+5.1%",  # Placeholder
            "ytd_return": "+18.7%",  # Placeholder
            "win_rate": f"{win_rate:.1f}%",
            "best_performer": sorted_by_score[0]["symbol"] if sorted_by_score else "N/A",
            "worst_performer": sorted_by_score[-1]["symbol"] if sorted_by_score else "N/A",
        }

    def _create_synthesis_prompt(self, symbol: str, llm_responses: Dict, latest_data: Dict) -> str:
        """Create comprehensive synthesis prompt using Jinja2 template with all quarterly analyses, comprehensive analysis, and technical data"""

        # 1. ORGANIZE FUNDAMENTAL DATA BY TYPE
        comprehensive_analysis = ""
        quarterly_analyses = []
        financial_metrics_by_quarter = []

        # Extract comprehensive analysis
        if "comprehensive" in llm_responses.get("fundamental", {}):
            comp_data = llm_responses["fundamental"]["comprehensive"]
            if comp_data and comp_data.get("content"):
                content = comp_data["content"]
                if isinstance(content, dict):
                    comprehensive_analysis = json.dumps(content, indent=2)
                else:
                    comprehensive_analysis = str(content)[:10000]

        # Extract and sort quarterly analyses chronologically
        for key, resp in llm_responses.get("fundamental", {}).items():
            if key != "comprehensive" and resp and resp.get("content"):
                period = resp.get("period", "Unknown")
                form_type = resp.get("form_type", "Unknown")
                content = resp.get("content", {})

                if isinstance(content, dict):
                    content_str = json.dumps(content, indent=2)[:3000]  # Limit for readability
                else:
                    content_str = str(content)[:3000]

                quarterly_analyses.append(
                    {
                        "period": period,
                        "form_type": form_type,
                        "content": content_str,
                        "raw_data": content if isinstance(content, dict) else {},
                    }
                )

                # Extract key financial metrics for trend analysis
                if isinstance(content, dict):
                    metrics = self._extract_financial_metrics_from_quarter(content, period)
                    if metrics:
                        financial_metrics_by_quarter.append(metrics)

        # Sort quarterly analyses chronologically (newest first)
        quarterly_analyses.sort(key=lambda x: x["period"], reverse=True)

        # 2. CREATE FINANCIAL TRENDS AND RATIOS
        financial_trends = self._create_financial_trends_analysis(financial_metrics_by_quarter)

        # 3. EXTRACT TECHNICAL ANALYSIS
        technical_analysis = ""
        technical_signals = {}

        if llm_responses.get("technical") and llm_responses["technical"].get("content"):
            technical_content = llm_responses["technical"]["content"]
            if isinstance(technical_content, dict):
                technical_analysis = json.dumps(technical_content, indent=2)
                technical_signals = technical_content
            else:
                technical_analysis = str(technical_content)[:6000]
                # Try to extract signals from text
                technical_signals = self._extract_technical_signals_from_text(technical_analysis)

        # 4. GET CURRENT MARKET DATA
        current_price = latest_data.get("technical", {}).get("current_price", 0)
        market_data = latest_data.get("technical", {})

        # 5. USE JINJA2 TEMPLATE TO CREATE SYNTHESIS PROMPT
        from investigator.application.prompts import get_prompt_manager

        prompt_manager = get_prompt_manager()

        template_data = {
            "symbol": symbol,
            "current_price": current_price,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "comprehensive_analysis": comprehensive_analysis,
            "quarterly_analyses": quarterly_analyses[:8],  # Limit to 8 most recent
            "quarterly_count": len(quarterly_analyses),
            "financial_trends": financial_trends,
            "technical_analysis": technical_analysis,
            "technical_signals": technical_signals,
            "market_data": market_data,
        }

        try:
            # Use appropriate template based on synthesis mode
            template_name = "investment_synthesis_comprehensive_mode.j2"
            prompt = prompt_manager.render_template(template_name, **template_data)
            self.main_logger.info(
                f"✅ Generated comprehensive synthesis prompt using Jinja2 template: {len(prompt)} chars, "
                f"{len(quarterly_analyses)} quarters, "
                f"{'✅' if comprehensive_analysis else '❌'} comprehensive, "
                f"{'✅' if technical_analysis else '❌'} technical"
            )
            return prompt

        except Exception as e:
            self.main_logger.error(f"❌ Failed to render Jinja2 comprehensive synthesis template: {e}")
            # Fallback to simple template
            return f"""Investment synthesis for {symbol} at ${current_price}:
Comprehensive: {'✅' if comprehensive_analysis else '❌'}
Quarterly: {len(quarterly_analyses)} quarters  
Technical: {'✅' if technical_analysis else '❌'}
Respond with detailed JSON investment analysis."""

    def _create_quarterly_synthesis_prompt(
        self, symbol: str, llm_responses: Dict, latest_data: Dict, prompt_manager
    ) -> str:
        """Create quarterly synthesis prompt using last N quarters + technical analysis (no comprehensive analysis)"""

        # Get symbol-specific logger
        symbol_logger = self.config.get_symbol_logger(symbol, "synthesizer")

        # 1. EXTRACT ALL QUARTERLY ANALYSES (NO COMPREHENSIVE)
        quarterly_analyses = []

        # Extract and sort quarterly analyses chronologically
        for key, resp in llm_responses.get("fundamental", {}).items():
            if key != "comprehensive" and resp and resp.get("content"):
                period = resp.get("period", "Unknown")
                form_type = resp.get("form_type", "Unknown")
                content = resp.get("content", {})

                if isinstance(content, dict):
                    content_str = json.dumps(content, indent=2)[:4000]  # More space since no comprehensive
                else:
                    content_str = str(content)[:4000]

                quarterly_analyses.append(
                    {
                        "period": period,
                        "form_type": form_type,
                        "content": content_str,
                        "raw_data": content if isinstance(content, dict) else {},
                    }
                )

        # Sort quarterly analyses by period (most recent first)
        quarterly_analyses.sort(key=lambda x: x["period"], reverse=True)
        quarterly_count = len(quarterly_analyses)

        symbol_logger.info(f"Quarterly synthesis: Using {quarterly_count} quarterly analyses for {symbol}")

        # 2. EXTRACT TECHNICAL ANALYSIS
        technical_analysis = ""
        technical_signals = {}

        if llm_responses.get("technical"):
            tech_content = llm_responses["technical"].get("content", "")
            if isinstance(tech_content, dict):
                technical_analysis = json.dumps(tech_content, indent=2)[:3000]
                technical_signals = tech_content
            else:
                technical_analysis = str(tech_content)[:3000]

        # 3. PREPARE FINANCIAL TRENDS FROM QUARTERLY DATA
        financial_trends = self._extract_quarterly_trends(quarterly_analyses)

        # 4. USE QUARTERLY-SPECIFIC SYNTHESIS TEMPLATE
        quarterly_synthesis_prompt = prompt_manager.render_template(
            "investment_synthesis_quarterly_mode.j2",
            symbol=symbol,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            current_price=latest_data.get("current_price", 0.0),
            comprehensive_analysis="",  # No comprehensive analysis in quarterly mode
            quarterly_analyses=quarterly_analyses,
            quarterly_count=quarterly_count,
            financial_trends=financial_trends,
            technical_analysis=technical_analysis,
            technical_signals=technical_signals,
            market_data=latest_data,
        )

        symbol_logger.info(f"Generated quarterly synthesis prompt: {len(quarterly_synthesis_prompt)} characters")
        return quarterly_synthesis_prompt

    def _extract_quarterly_trends(self, quarterly_analyses: List[Dict]) -> str:
        """Extract and summarize trends across quarters for quarterly synthesis"""
        if not quarterly_analyses:
            return "No quarterly data available for trend analysis"

        trends = []
        trends.append(f"Quarterly Analysis Summary ({len(quarterly_analyses)} quarters):")

        # Add trends based on available quarterly data
        for i, qa in enumerate(quarterly_analyses[:8]):  # Use last 8 quarters max
            period = qa.get("period", f"Q{i+1}")
            form_type = qa.get("form_type", "Unknown")
            trends.append(f"- {period} ({form_type}): Key financial metrics and performance indicators")

        if len(quarterly_analyses) > 8:
            trends.append(f"... and {len(quarterly_analyses) - 8} additional quarters")

        return "\n".join(trends)

    def _extract_financial_metrics_from_quarter(self, quarter_data: Dict, period: str) -> Optional[Dict]:
        """Extract key financial metrics from a quarterly analysis"""
        try:
            metrics = {"period": period}

            # Try to extract common financial metrics from the quarter data
            if isinstance(quarter_data, dict):
                # Look for revenue metrics
                for key in ["revenue", "total_revenue", "revenues", "sales"]:
                    if key in quarter_data:
                        metrics["revenue"] = quarter_data[key]
                        break

                # Look for profit metrics
                for key in ["net_income", "net_profit", "earnings", "profit"]:
                    if key in quarter_data:
                        metrics["net_income"] = quarter_data[key]
                        break

                # Look for margin metrics
                for key in ["gross_margin", "operating_margin", "profit_margin"]:
                    if key in quarter_data:
                        metrics[key] = quarter_data[key]

                # Look for other key metrics
                for key in ["eps", "operating_cash_flow", "free_cash_flow", "total_assets", "total_debt"]:
                    if key in quarter_data:
                        metrics[key] = quarter_data[key]

            return metrics if len(metrics) > 1 else None

        except Exception as e:
            self.main_logger.warning(f"Error extracting metrics from quarter {period}: {e}")
            return None

    def _create_financial_trends_analysis(self, metrics_by_quarter: List[Dict]) -> str:
        """Create financial trends analysis from quarterly metrics"""
        try:
            if not metrics_by_quarter:
                return "[NO QUARTERLY METRICS AVAILABLE FOR TREND ANALYSIS]"

            trends = []
            trends.append(f"📊 FINANCIAL TRENDS ANALYSIS ({len(metrics_by_quarter)} quarters):")

            # Sort by period for chronological analysis
            sorted_metrics = sorted(metrics_by_quarter, key=lambda x: x.get("period", ""))

            # Revenue trend
            revenues = [m.get("revenue", 0) for m in sorted_metrics if m.get("revenue")]
            if len(revenues) >= 2:
                revenue_growth = ((revenues[-1] - revenues[0]) / revenues[0] * 100) if revenues[0] > 0 else 0
                trends.append(f"📈 Revenue Trend: {revenue_growth:+.1f}% over {len(revenues)} quarters")

            # Margin trends
            margins = [m.get("profit_margin", 0) for m in sorted_metrics if m.get("profit_margin")]
            if len(margins) >= 2:
                margin_change = margins[-1] - margins[0]
                trends.append(f"💰 Margin Trend: {margin_change:+.1f}pp change in profit margin")

            # Add quarterly breakdown
            trends.append("\n📋 Quarterly Progression:")
            for i, metrics in enumerate(sorted_metrics[-4:]):  # Last 4 quarters
                period = metrics.get("period", f"Q{i+1}")
                revenue = metrics.get("revenue", 0)
                margin = metrics.get("profit_margin", 0)
                trends.append(f"  {period}: Revenue ${revenue:,.0f}M, Margin {margin:.1f}%")

            return "\n".join(trends)

        except Exception as e:
            self.main_logger.warning(f"Error creating trends analysis: {e}")
            return "[ERROR CREATING TRENDS ANALYSIS]"

    def _extract_technical_signals_from_text(self, technical_text: str) -> Dict:
        """Extract technical signals from text analysis"""
        try:
            import re

            signals = {}

            # Extract RSI
            rsi_match = re.search(r"RSI[^:]*:\s*([\d.]+)", technical_text, re.IGNORECASE)
            if rsi_match:
                signals["rsi"] = float(rsi_match.group(1))

            # Extract MACD
            macd_match = re.search(r"MACD[^:]*:\s*([-\d.]+)", technical_text, re.IGNORECASE)
            if macd_match:
                signals["macd"] = float(macd_match.group(1))

            # Extract trend
            trend_match = re.search(r"trend[^:]*:\s*([A-Za-z]+)", technical_text, re.IGNORECASE)
            if trend_match:
                signals["trend"] = trend_match.group(1).upper()

            # Extract support/resistance
            support_match = re.search(r"support[^:]*:\s*\$?([\d.]+)", technical_text, re.IGNORECASE)
            if support_match:
                signals["support"] = float(support_match.group(1))

            resistance_match = re.search(r"resistance[^:]*:\s*\$?([\d.]+)", technical_text, re.IGNORECASE)
            if resistance_match:
                signals["resistance"] = float(resistance_match.group(1))

            return signals

        except Exception as e:
            self.main_logger.warning(f"Error extracting technical signals: {e}")
            return {}

    def _get_sector_context(self, symbol: str) -> str:
        """Get sector and industry context for the symbol"""
        try:
            # Load sector mapping from external file
            sector_mapping_file = Path(self.config.data_dir) / "sector_mapping.json"

            if sector_mapping_file.exists():
                with open(sector_mapping_file, "r") as f:
                    sector_data = json.load(f)

                mappings = sector_data.get("sector_mappings", {})
                default = sector_data.get("default_mapping", {})

                if symbol in mappings:
                    mapping = mappings[symbol]
                    return f"{mapping['sector']} - {mapping['industry']}"
                else:
                    return f"{default['sector']} - {default['industry']}"
            else:
                self.main_logger.warning(f"Sector mapping file not found: {sector_mapping_file}")
                return "Unknown Sector - Requires Research"

        except Exception as e:
            self.main_logger.error(f"Error loading sector mapping: {e}")
            return "Unknown Sector - Requires Research"

    def _get_market_environment_context(self) -> str:
        """Get current market environment context"""
        # This could be enhanced to fetch real market data
        return "Mixed signals with elevated volatility, Fed policy uncertainty, and sector rotation dynamics"

    def _calculate_ma_position(self, current_price: float, ma_price: float) -> str:
        """Calculate moving average position relative to current price"""
        if not current_price or not ma_price:
            return "N/A"

        if current_price > ma_price * 1.02:
            return "Strong Above"
        elif current_price > ma_price:
            return "Above"
        elif current_price < ma_price * 0.98:
            return "Strong Below"
        else:
            return "Below"

    def _check_ma_cross(self, sma_50: float, sma_200: float) -> str:
        """Check for golden/death cross pattern"""
        if not sma_50 or not sma_200:
            return "N/A"

        if sma_50 > sma_200 * 1.01:
            return "Golden Cross"
        elif sma_50 < sma_200 * 0.99:
            return "Death Cross"
        else:
            return "Neutral"

    def _assess_trend_strength(self, tech_data: Dict) -> str:
        """Assess overall trend strength from technical data"""
        try:
            rsi = tech_data.get("rsi", 50)
            price_change_1m = tech_data.get("price_change_1m", 0)

            if rsi > 60 and price_change_1m > 5:
                return "Strong Bullish"
            elif rsi > 50 and price_change_1m > 0:
                return "Bullish"
            elif rsi < 40 and price_change_1m < -5:
                return "Strong Bearish"
            elif rsi < 50 and price_change_1m < 0:
                return "Bearish"
            else:
                return "Neutral"
        except:
            return "N/A"

    def _calculate_bb_position(self, tech_data: Dict) -> str:
        """Calculate Bollinger Band position"""
        try:
            current_price = tech_data.get("current_price", 0)
            bb_upper = tech_data.get("bollinger_upper", 0)
            bb_lower = tech_data.get("bollinger_lower", 0)

            if not all([current_price, bb_upper, bb_lower]):
                return "N/A"

            bb_range = bb_upper - bb_lower
            position = (current_price - bb_lower) / bb_range

            if position > 0.8:
                return "Upper Band"
            elif position > 0.6:
                return "Above Middle"
            elif position > 0.4:
                return "Middle Range"
            elif position > 0.2:
                return "Below Middle"
            else:
                return "Lower Band"
        except:
            return "N/A"

    def _assess_volume_trend(self, tech_data: Dict) -> str:
        """Assess volume trend"""
        try:
            volume_ratio = tech_data.get("volume_ratio", 1)

            if volume_ratio > 2.0:
                return "Very High"
            elif volume_ratio > 1.5:
                return "High"
            elif volume_ratio > 0.8:
                return "Normal"
            elif volume_ratio > 0.5:
                return "Low"
            else:
                return "Very Low"
        except:
            return "N/A"

    def _assess_volume_price_relationship(self, tech_data: Dict) -> str:
        """Assess volume-price relationship"""
        try:
            price_change_1d = tech_data.get("price_change_1d", 0)
            volume_ratio = tech_data.get("volume_ratio", 1)

            if price_change_1d > 0 and volume_ratio > 1.2:
                return "Bullish Confirmation"
            elif price_change_1d < 0 and volume_ratio > 1.2:
                return "Bearish Confirmation"
            elif abs(price_change_1d) > 2 and volume_ratio < 0.8:
                return "Divergence Warning"
            else:
                return "Neutral"
        except:
            return "N/A"

    def _create_fallback_recommendation(self, raw_response: Any, symbol: str, overall_score: float) -> Dict[str, Any]:
        """
        Create a fallback recommendation when JSON parsing fails

        Args:
            raw_response: The raw LLM response that failed to parse
            symbol: Stock symbol
            overall_score: Computed overall score

        Returns:
            Dict containing fallback recommendation structure
        """
        try:
            # Convert response to string for text parsing
            response_text = str(raw_response) if raw_response else ""

            # Try to extract any partial information using regex
            import re

            # Extract recommendation if present
            recommendation = "HOLD"  # Safe default
            rec_patterns = [
                r'recommendation["\']?\s*:\s*["\']?(STRONG_BUY|STRONG_SELL|BUY|SELL|HOLD)["\']?',
                r"FINAL\s+RECOMMENDATION[:\s]*\*?\*?\s*\[?([A-Z\s]+)\]?",
                r'"recommendation":\s*"([^"]+)"',
            ]

            for pattern in rec_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    rec_text = match.group(1).strip().upper()
                    if any(valid in rec_text for valid in ["BUY", "SELL", "HOLD"]):
                        recommendation = rec_text
                        break

            # Extract confidence if present
            confidence = "LOW"  # Conservative default for failed parsing
            conf_patterns = [
                r'confidence["\']?\s*:\s*["\']?(HIGH|MEDIUM|LOW)["\']?',
                r'"confidence_level":\s*"([^"]+)"',
            ]

            for pattern in conf_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    confidence = match.group(1).strip().upper()
                    break

            # Extract investment thesis if present
            thesis = f"Analysis completed for {symbol} with computed overall score of {overall_score:.1f}/10."
            thesis_patterns = [
                r'investment_thesis["\']?\s*:\s*["\']([^"\']+)["\']',
                r'thesis["\']?\s*:\s*["\']([^"\']+)["\']',
                r"INVESTMENT\s+THESIS[:\s]*([^{}\[\]]+?)(?=\*\*|##|\n\n|$)",
            ]

            for pattern in thesis_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    extracted_thesis = match.group(1).strip()
                    if len(extracted_thesis) > 20:  # Only use if substantial
                        thesis = extracted_thesis[:500]  # Limit length
                        break

            # Create fallback structure that matches expected format
            fallback_recommendation = {
                "overall_score": overall_score,
                "fundamental_score": overall_score,  # Use computed score as fallback
                "technical_score": overall_score,  # Use computed score as fallback
                "investment_recommendation": {"recommendation": recommendation, "confidence_level": confidence},
                "executive_summary": {"investment_thesis": thesis},
                "key_catalysts": [
                    f"Technical and fundamental analysis for {symbol}",
                    "Market position assessment",
                    "Financial performance review",
                ],
                "key_risks": [
                    "JSON parsing failure indicates potential data quality issues",
                    "LLM response formatting problems",
                    "Analysis may be incomplete due to parsing errors",
                ],
                "position_size": "SMALL",  # Conservative due to parsing failure
                "time_horizon": "MEDIUM-TERM",
                "entry_strategy": f"Conservative approach recommended due to analysis parsing issues",
                "exit_strategy": f"Monitor for improved data quality and re-analyze",
                "details": f"Fallback recommendation created due to JSON parsing failure. Raw response length: {len(response_text)} characters.",
                "_fallback_created": True,  # Flag to indicate this is a fallback
                "_parsing_error": True,  # Flag to indicate parsing issues
            }

            self.main_logger.info(
                f"Created fallback recommendation for {symbol}: {recommendation} (confidence: {confidence})"
            )

            return fallback_recommendation

        except Exception as e:
            self.main_logger.error(f"Error creating fallback recommendation: {e}")
            # Last resort fallback
            return {
                "overall_score": 5.0,  # Neutral score
                "fundamental_score": 5.0,
                "technical_score": 5.0,
                "investment_recommendation": {"recommendation": "HOLD", "confidence_level": "LOW"},
                "executive_summary": {
                    "investment_thesis": f"Unable to complete analysis for {symbol} due to processing errors."
                },
                "key_catalysts": ["Analysis pending"],
                "key_risks": ["Analysis incomplete", "Data processing errors"],
                "position_size": "AVOID",
                "time_horizon": "UNKNOWN",
                "entry_strategy": "Wait for successful analysis",
                "exit_strategy": "Not applicable",
                "details": "Emergency fallback due to complete parsing failure",
                "_fallback_created": True,
                "_parsing_error": True,
                "_emergency_fallback": True,
            }

    def _create_extensible_insights_structure(
        self, ai_recommendation: Dict, thinking_content: str, additional_details: str, symbol: str
    ) -> Dict[str, Any]:
        """
        Create an extensible structure for capturing additional insights that can evolve with
        prompt and response changes. This structure is designed to be included in the final
        PDF report so reporting and synthesizing modules can inspect this key to cover fields
        graciously.

        Args:
            ai_recommendation: The main AI recommendation dict
            thinking_content: LLM thinking/reasoning content
            additional_details: Any additional text details from response
            symbol: Stock symbol being analyzed

        Returns:
            Dict containing extensible insights structure
        """
        try:
            extensible_insights = {
                # Core metadata about the insights capture
                "_insights_version": "1.0",
                "_symbol": symbol,
                "_timestamp": datetime.now().isoformat(),
                "_capture_method": "llm_response_analysis",
                # Structured thinking and reasoning capture
                "reasoning_insights": {
                    "thinking_content": thinking_content if thinking_content else "",
                    "thinking_length": len(thinking_content) if thinking_content else 0,
                    "has_structured_reasoning": bool(thinking_content and len(thinking_content) > 100),
                    "reasoning_themes": self._extract_reasoning_themes(thinking_content) if thinking_content else [],
                    "decision_process": self._extract_decision_process(thinking_content) if thinking_content else {},
                },
                # Additional content and markdown capture
                "content_insights": {
                    "additional_details": additional_details if additional_details else "",
                    "details_length": len(additional_details) if additional_details else 0,
                    "has_markdown_content": (
                        self._detect_markdown_content(additional_details) if additional_details else False
                    ),
                    "extracted_bullet_points": (
                        self._extract_bullet_points(additional_details) if additional_details else []
                    ),
                    "extracted_numbers": (
                        self._extract_numerical_insights(additional_details) if additional_details else []
                    ),
                },
                # Response structure analysis
                "response_structure": {
                    "field_completeness": self._analyze_field_completeness(ai_recommendation),
                    "response_type": type(ai_recommendation).__name__,
                    "custom_fields": self._identify_custom_fields(ai_recommendation),
                    "processing_metadata": ai_recommendation.get("processing_metadata", {}),
                    "contains_fallback_flags": self._check_fallback_flags(ai_recommendation),
                },
                # Future evolution placeholders (for prompt/response evolution)
                "evolution_capture": {
                    "sentiment_analysis": {},  # Future: sentiment from thinking
                    "confidence_indicators": {},  # Future: confidence markers in text
                    "methodology_insights": {},  # Future: LLM methodology preferences
                    "risk_assessment_depth": {},  # Future: detailed risk analysis
                    "peer_comparison_insights": {},  # Future: peer analysis reasoning
                    "market_timing_insights": {},  # Future: timing-specific insights
                    "scenario_analysis": {},  # Future: what-if scenario reasoning
                },
                # Report integration guidance
                "report_integration": {
                    "should_include_thinking": bool(thinking_content and len(thinking_content) > 200),
                    "should_include_details": bool(additional_details and len(additional_details) > 50),
                    "recommended_report_sections": self._recommend_report_sections(
                        ai_recommendation, thinking_content, additional_details
                    ),
                    "priority_insights": self._extract_priority_insights(thinking_content, additional_details),
                    "visualization_suggestions": self._suggest_visualizations(ai_recommendation),
                },
            }

            return extensible_insights

        except Exception as e:
            self.main_logger.error(f"Error creating extensible insights structure: {e}")
            # Return minimal structure on error
            return {
                "_insights_version": "1.0",
                "_symbol": symbol,
                "_timestamp": datetime.now().isoformat(),
                "_error": str(e),
                "reasoning_insights": {},
                "content_insights": {},
                "response_structure": {},
                "evolution_capture": {},
                "report_integration": {"should_include_thinking": False, "should_include_details": False},
            }

    def _extract_reasoning_themes(self, thinking_content: str) -> List[str]:
        """Extract key reasoning themes from thinking content"""
        if not thinking_content:
            return []

        themes = []
        # Look for common reasoning patterns
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
                # Extract the matched theme
                match = re.search(pattern, thinking_content, re.IGNORECASE)
                if match:
                    themes.append(match.group(0).lower())

        return list(set(themes))  # Remove duplicates

    def _extract_decision_process(self, thinking_content: str) -> Dict[str, Any]:
        """Extract decision-making process from thinking content"""
        if not thinking_content:
            return {}

        process = {
            "has_structured_approach": bool(re.search(r"first|then|next|finally", thinking_content, re.IGNORECASE)),
            "considers_alternatives": bool(
                re.search(r"but|however|alternatively|on the other hand", thinking_content, re.IGNORECASE)
            ),
            "weighs_factors": bool(re.search(r"weight|balance|consider|factor", thinking_content, re.IGNORECASE)),
            "mentions_uncertainty": bool(
                re.search(r"uncertain|unclear|maybe|might|could", thinking_content, re.IGNORECASE)
            ),
            "shows_confidence": bool(re.search(r"confident|certain|sure|clear", thinking_content, re.IGNORECASE)),
        }

        return process

    def _detect_markdown_content(self, content: str) -> bool:
        """Detect if content contains markdown formatting"""
        if not content:
            return False

        markdown_patterns = [r"\*\*.*?\*\*", r"\*.*?\*", r"#+ ", r"- ", r"\d+\. ", r"```"]
        return any(re.search(pattern, content) for pattern in markdown_patterns)

    def _extract_bullet_points(self, content: str) -> List[str]:
        """Extract bullet points from content"""
        if not content:
            return []

        # Look for various bullet point patterns
        patterns = [r"- (.+)", r"• (.+)", r"\* (.+)", r"\d+\. (.+)"]
        bullets = []

        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            bullets.extend(matches)

        return [bullet.strip() for bullet in bullets if len(bullet.strip()) > 5]

    def _extract_numerical_insights(self, content: str) -> List[Dict[str, Any]]:
        """Extract numerical insights from content"""
        if not content:
            return []

        numbers = []
        # Look for percentages, ratios, monetary amounts
        patterns = [
            (r"(\d+(?:\.\d+)?%)", "percentage"),
            (r"\$(\d+(?:,\d{3})*(?:\.\d{2})?)", "monetary"),
            (r"(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", "ratio"),
            (r"(\d+(?:\.\d+)?x)", "multiple"),
        ]

        for pattern, num_type in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    numbers.append({"type": num_type, "value": match, "context": "content_extraction"})
                else:
                    numbers.append({"type": num_type, "value": match, "context": "content_extraction"})

        return numbers[:10]  # Limit to first 10 to avoid clutter

    def _analyze_field_completeness(self, ai_recommendation: Dict) -> Dict[str, Any]:
        """Analyze completeness of standard fields in AI recommendation"""
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

    def _identify_custom_fields(self, ai_recommendation: Dict) -> List[str]:
        """Identify custom/non-standard fields in the response"""
        standard_fields = {
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

        custom_fields = []
        for key in ai_recommendation.keys():
            if key not in standard_fields:
                custom_fields.append(key)

        return custom_fields

    def _check_fallback_flags(self, ai_recommendation: Dict) -> Dict[str, bool]:
        """Check for various fallback and error flags"""
        return {
            "is_fallback": ai_recommendation.get("_fallback_created", False),
            "has_parsing_error": ai_recommendation.get("_parsing_error", False),
            "is_emergency_fallback": ai_recommendation.get("_emergency_fallback", False),
            "has_processing_metadata": bool(ai_recommendation.get("processing_metadata")),
        }

    def _recommend_report_sections(
        self, ai_recommendation: Dict, thinking_content: str, additional_details: str
    ) -> List[str]:
        """Recommend which report sections should be included based on available content"""
        sections = ["executive_summary", "recommendation"]  # Always include these

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

    def _extract_priority_insights(self, thinking_content: str, additional_details: str) -> List[str]:
        """Extract the most important insights for highlighting in reports"""
        insights = []

        if thinking_content:
            # Look for key insights in thinking
            key_phrases = re.findall(
                r"(?:key|important|crucial|critical|significant).{1,100}", thinking_content, re.IGNORECASE
            )
            insights.extend([phrase.strip() for phrase in key_phrases[:3]])

        if additional_details:
            # Look for highlighted items in details
            highlights = re.findall(r"(?:\*\*|##).{1,100}", additional_details)
            insights.extend([highlight.strip() for highlight in highlights[:2]])

        return insights

    def _suggest_visualizations(self, ai_recommendation: Dict) -> List[str]:
        """Suggest visualizations based on the content of the recommendation"""
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


def main():
    """Main entry point for standalone synthesis"""
    import argparse

    # Display synthesis banner
    ASCIIArt.print_banner("synthesis")

    # Get main logger for the standalone execution
    config = get_config()
    main_logger = config.get_main_logger("synthesizer_main")

    parser = argparse.ArgumentParser(description="Investment Synthesizer")
    parser.add_argument("--symbol", help="Stock symbol to analyze (required unless --weekly)")
    parser.add_argument("--symbols", nargs="*", help="Multiple stock symbols for batch analysis")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--report", action="store_true", help="Generate PDF report")
    parser.add_argument("--weekly", action="store_true", help="Generate weekly report")
    parser.add_argument("--send-email", action="store_true", help="Send report via email")
    parser.add_argument(
        "--synthesis-mode",
        choices=["comprehensive", "quarterly"],
        default="comprehensive",
        help="Synthesis approach: comprehensive (default) or quarterly",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.weekly and not args.symbol and not args.symbols:
        parser.error("Either --symbol, --symbols, or --weekly is required")

    synthesizer = InvestmentSynthesizer()

    if args.weekly:
        # Generate weekly report for all tracked stocks
        try:
            tracked_stocks = config.stocks_to_track
            print(f"📊 Generating weekly report for {len(tracked_stocks)} stocks...")

            recommendations = []
            for symbol in tracked_stocks:
                try:
                    print(f"  Synthesizing analysis for {symbol}...")
                    recommendation = synthesizer.synthesize_analysis(symbol)
                    recommendations.append(recommendation)
                    print(f"  ✅ {symbol}: {recommendation.recommendation} ({recommendation.overall_score:.1f}/10)")
                except Exception as e:
                    print(f"  ❌ {symbol}: Failed to synthesize - {e}")
                    main_logger.warning(f"Failed to synthesize {symbol}: {e}")

            if recommendations:
                report_path = synthesizer.generate_report(recommendations, "weekly")
                print(f"\n📊 Weekly report generated: {report_path}")

                if args.send_email:
                    # TODO: Implement email sending
                    print("📧 Email sending not yet implemented")
            else:
                print("❌ No successful recommendations to include in weekly report")
                return 1

        except Exception as e:
            print(f"❌ Weekly report generation failed: {e}")
            main_logger.error(f"Weekly report generation failed: {e}")
            return 1

    elif args.symbols:
        # Batch analysis for multiple symbols
        recommendations = []
        for symbol in args.symbols:
            try:
                print(f"Synthesizing analysis for {symbol}...")
                recommendation = synthesizer.synthesize_analysis(symbol, args.synthesis_mode)
                recommendations.append(recommendation)
                print(f"✅ {symbol}: {recommendation.recommendation} ({recommendation.overall_score:.1f}/10)")
            except Exception as e:
                print(f"❌ {symbol}: Failed to synthesize - {e}")

        if args.report and recommendations:
            report_path = synthesizer.generate_report(recommendations, "batch")
            print(f"\n📊 Batch report generated: {report_path}")

    else:
        # Single symbol analysis
        symbol = args.symbol.upper()
        try:
            recommendation = synthesizer.synthesize_analysis(symbol, args.synthesis_mode)

            # Print summary
            print(f"\n{'='*60}")
            print(f"Investment Recommendation for {symbol}")
            print(f"{'='*60}")
            print(f"Overall Score: {recommendation.overall_score:.1f}/10")
            print(f"├─ Fundamental: {recommendation.fundamental_score:.1f}/10")
            print(f"│  ├─ Income Statement: {recommendation.income_score:.1f}/10")
            print(f"│  ├─ Cash Flow: {recommendation.cashflow_score:.1f}/10")
            print(f"│  ├─ Balance Sheet: {recommendation.balance_score:.1f}/10")
            print(f"│  ├─ Growth Score: {recommendation.growth_score:.1f}/10")
            print(f"│  ├─ Value Score: {recommendation.value_score:.1f}/10")
            print(f"│  ├─ Business Quality: {recommendation.business_quality_score:.1f}/10")
            print(f"│  └─ Data Quality: {recommendation.data_quality_score:.1f}/10")
            print(f"└─ Technical: {recommendation.technical_score:.1f}/10")
            print(f"\nRecommendation: {recommendation.recommendation}")
            print(f"Confidence: {recommendation.confidence}")
            print(f"Time Horizon: {recommendation.time_horizon}")
            print(f"Position Size: {recommendation.position_size}")

            if recommendation.price_target:
                print(f"\nPrice Target: ${recommendation.price_target:.2f}")
                print(f"Current Price: ${recommendation.current_price:.2f}")
                upside = (
                    ((recommendation.price_target / recommendation.current_price - 1) * 100)
                    if recommendation.current_price > 0
                    else 0
                )
                print(f"Upside Potential: {upside:+.1f}%")

            # Generate report if requested
            if args.report:
                report_path = synthesizer.generate_report([recommendation], "synthesis")
                print(f"\n📊 Report generated: {report_path}")

        except Exception as e:
            print(f"❌ Analysis failed for {symbol}: {e}")
            main_logger.error(f"Analysis failed for {symbol}: {e}")
            return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
