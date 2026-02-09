"""
ETF Market Context Agent
Provides market and sector context using ETF data from database
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from investigator.domain.agents.base import AgentResult, AgentTask, AnalysisType, InvestmentAgent, TaskStatus
from investigator.infrastructure.database.market_data import get_market_data_fetcher  # Uses singleton pattern
from investigator.infrastructure.external.fred import (
    MacroIndicatorsFetcher,
    format_indicator_for_display,
)

logger = logging.getLogger(__name__)


@dataclass
class SectorMapping:
    """Maps sectors to their representative ETFs"""

    sector: str
    primary_etf: str
    secondary_etfs: List[str]
    description: str


class ETFMarketContextAgent(InvestmentAgent):
    """
    Agent that provides market and sector context using ETF data

    Uses ETF performance to assess:
    - Overall market sentiment (SPY)
    - Sector performance vs market
    - Relative strength analysis
    - Risk-on vs risk-off sentiment
    - Sector rotation signals
    """

    def __init__(self, agent_id: str, ollama_client, event_bus, cache_manager):
        from investigator.config import get_config

        self.config = get_config()
        self.market_context_model = self.config.ollama.models.get(
            "market_context", self.config.ollama.models.get("synthesis", "deepseek-r1:32b")
        )

        super().__init__(agent_id, ollama_client, event_bus, cache_manager)
        self.analysis_type = AnalysisType.MARKET_CONTEXT

        # Initialize market data fetcher (uses singleton)
        self.market_data_fetcher = get_market_data_fetcher(self.config)

        # Initialize macro indicators fetcher
        self.macro_fetcher = MacroIndicatorsFetcher()

        # Define sector ETF mappings
        self.sector_etfs = {
            "technology": SectorMapping(
                "technology",
                "XLK",
                [],  # Removed secondary ETFs - primary is sufficient
                "Technology Select Sector SPDR Fund",
            ),
            "healthcare": SectorMapping(
                "healthcare",
                "XLV",
                [],  # Removed secondary ETFs - primary is sufficient
                "Health Care Select Sector SPDR Fund",
            ),
            "financials": SectorMapping(
                "financials",
                "XLF",
                [],  # Removed secondary ETFs - primary is sufficient
                "Financial Select Sector SPDR Fund",
            ),
            "energy": SectorMapping(
                "energy", "XLE", [], "Energy Select Sector SPDR Fund"  # Removed secondary ETFs - primary is sufficient
            ),
            "industrials": SectorMapping(
                "industrials",
                "XLI",
                [],  # Removed secondary ETFs (VIS, IYJ, FXI) - primary is sufficient
                "Industrial Select Sector SPDR Fund",
            ),
            "consumer_discretionary": SectorMapping(
                "consumer_discretionary",
                "XLY",
                [],  # Removed secondary ETFs - primary is sufficient
                "Consumer Discretionary Select Sector SPDR Fund",
            ),
            "consumer_staples": SectorMapping(
                "consumer_staples",
                "XLP",
                [],  # Removed secondary ETFs - primary is sufficient
                "Consumer Staples Select Sector SPDR Fund",
            ),
            "utilities": SectorMapping(
                "utilities",
                "XLU",
                [],  # Removed secondary ETFs - primary is sufficient
                "Utilities Select Sector SPDR Fund",
            ),
            "materials": SectorMapping(
                "materials",
                "XLB",
                [],  # Removed secondary ETFs - primary is sufficient
                "Materials Select Sector SPDR Fund",
            ),
            "real_estate": SectorMapping(
                "real_estate",
                "XLRE",
                [],  # Removed secondary ETFs - primary is sufficient
                "Real Estate Select Sector SPDR Fund",
            ),
            "communication": SectorMapping(
                "communication",
                "XLC",
                [],  # Removed secondary ETFs - primary is sufficient
                "Communication Services Select Sector SPDR Fund",
            ),
        }

        # Market benchmarks
        self.market_etfs = {
            "broad_market": "SPY",  # S&P 500
            "nasdaq": "QQQ",  # Nasdaq 100
            "small_cap": "IWM",  # Russell 2000
            "international": "EFA",  # Developed Markets
            "emerging": "EEM",  # Emerging Markets
            "bonds": "AGG",  # Bond Market
            "treasury": "TLT",  # 20+ Year Treasury
            "high_yield": "HYG",  # High Yield Corporate
            # 'volatility': VIX removed - now fetched from VIXCLS macro indicator (FRED database)
            "gold": "GLD",  # Gold
            "silver": "SLV",  # Silver
            "oil": "USO",  # Oil
            # 'agriculture': 'DBA',     # Agriculture Commodities (data ends 2023-04-06)
            "commodities": "DBC",  # Broad Commodities Index
            "commodities_alt": "GSG",  # S&P GSCI Commodity Index (energy-heavy)
        }

        # Define timeframes for analysis (Industry-Standard Trading Days)
        # These periods filter noise while capturing meaningful trends
        self.timeframes = {
            "leading": 10,  # ~2 weeks - Early warning indicator, high responsiveness
            "short_term": 21,  # 1 month - PRIMARY signal for sentiment (industry standard)
            "medium_term": 63,  # 3 months (quarter) - Trend confirmation, earnings-aligned
            "long_term": 252,  # 1 year - Strategic positioning baseline
        }

        # Timeframe metadata for reporting and cache storage
        self.timeframe_metadata = {
            "leading": {
                "label": "Leading (10d)",
                "description": "Early warning indicator - High noise, early signals",
                "use_case": "Tactical positioning and divergence detection",
                "reliability": "low",
            },
            "short_term": {
                "label": "Short-term (21d)",
                "description": "Primary sentiment signal - Industry standard monthly period",
                "use_case": "Main market sentiment and tactical trading decisions",
                "reliability": "high",
            },
            "medium_term": {
                "label": "Medium-term (63d)",
                "description": "Quarterly trend confirmation - Earnings cycle aligned",
                "use_case": "Portfolio rebalancing and trend validation",
                "reliability": "high",
            },
            "long_term": {
                "label": "Long-term (252d)",
                "description": "Strategic baseline - Full trading year perspective",
                "use_case": "Strategic positioning and fundamental trends",
                "reliability": "very_high",
            },
        }

    def register_capabilities(self) -> List:
        """Register agent capabilities"""
        from investigator.domain.agents.base import AgentCapability

        return [
            AgentCapability(
                analysis_type=AnalysisType.MARKET_CONTEXT,
                min_data_required={"symbol": str},
                max_processing_time=180,  # ETF data fetching + LLM analysis
                required_models=[self.market_context_model],
                cache_ttl=900,  # 15 minutes cache for market context
            )
        ]

    def _get_stock_data_cached(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Get stock data with caching

        Args:
            symbol: Stock symbol to fetch
            days: Number of days of historical data

        Returns:
            DataFrame with stock data
        """
        try:
            # Fetch from database (market_data_fetcher has its own caching)
            data = self.market_data_fetcher.get_stock_data(symbol, days)

            if data is None or data.empty:
                logger.warning(f"No data returned for {symbol} from market data fetcher")
                return pd.DataFrame()

            return data

        except Exception as e:
            logger.error(f"Failed to fetch stock data for {symbol}: {e}")
            return pd.DataFrame()

    def _get_macro_indicators_cached(self) -> Dict:
        """
        Get macro indicators with caching

        Cache TTL: 24 hours (macro indicators change slowly)
        Cache key: Global scope (shared across ALL stocks for maximum reusability)
        Cache type: MARKET_CONTEXT (not stock-specific)

        Returns:
            Dict with macro summary and Buffett indicator
        """
        from investigator.infrastructure.cache.cache_types import CacheType

        # Global cache key - shared across ALL stock analyses
        cache_key = {
            "scope": "global",  # Not stock-specific
            "data_type": "macro_indicators",  # Type of market data
            "date": datetime.now().strftime("%Y-%m-%d"),  # Daily refresh
        }

        try:
            # Try cache first
            cached_data = self.cache.get(CacheType.MARKET_CONTEXT, cache_key)

            if cached_data:
                logger.debug("Using cached macro indicators (global scope)")
                return cached_data

            # Fetch fresh data
            logger.info("Fetching fresh macro indicators from database (will cache globally)")

            macro_summary = self.macro_fetcher.get_macro_summary()
            buffett = self.macro_fetcher.calculate_buffett_indicator()

            result = {
                "macro_summary": macro_summary,
                "buffett_indicator": buffett,
                "fetched_at": datetime.now().isoformat(),
            }

            # Cache with MARKET_CONTEXT type - reusable across all stocks
            self.cache.set(CacheType.MARKET_CONTEXT, cache_key, result)

            logger.info("Macro indicators cached globally (reusable for all stocks today)")

            return result

        except Exception as e:
            logger.error(f"Failed to fetch macro indicators: {e}")
            return {"macro_summary": None, "buffett_indicator": None, "error": str(e)}

    async def process(self, task: AgentTask) -> AgentResult:
        """
        Process ETF market context analysis

        Args:
            task: Agent task with symbol and context

        Returns:
            AgentResult with market context analysis
        """
        start_time = datetime.now()
        symbol = task.symbol

        try:
            logger.info(f"Starting ETF market context analysis for {symbol}")

            # Determine sector for the symbol
            sector = self._determine_symbol_sector(symbol, task.context)

            # Phase 2: Check for pre-fetched consolidated data from DataSourceManager
            consolidated_data = task.context.get("consolidated_data")
            macro_indicators = None

            if consolidated_data is not None:
                try:
                    # Extract macro data from consolidated data
                    macro_data = getattr(consolidated_data, "macro", None) or (
                        consolidated_data.get("macro") if isinstance(consolidated_data, dict) else None
                    )
                    fed_data = getattr(consolidated_data, "fed_districts", None) or (
                        consolidated_data.get("fed_districts") if isinstance(consolidated_data, dict) else None
                    )
                    volatility_data = getattr(consolidated_data, "volatility", None) or (
                        consolidated_data.get("volatility") if isinstance(consolidated_data, dict) else None
                    )

                    if macro_data or fed_data or volatility_data:
                        macro_indicators = {
                            "macro_summary": macro_data,
                            "fed_districts": fed_data,
                            "volatility": volatility_data,
                            "buffett_indicator": None,  # Will be calculated if needed
                            "fetched_at": datetime.now().isoformat(),
                        }
                        logger.debug(f"Using pre-fetched macro data for {symbol} from DataSourceManager")
                except Exception as e:
                    logger.debug(f"Could not use consolidated macro data for {symbol}: {e}")

            # Fallback to legacy macro fetch if no pre-fetched data
            if macro_indicators is None:
                macro_indicators = self._get_macro_indicators_cached()

            # Get market context data (now includes VIX from macro_indicators)
            market_context = await self._analyze_market_context(macro_indicators)

            # Get sector context if sector is identified
            sector_context = {}
            if sector:
                sector_context = await self._analyze_sector_context(sector)

            # Analyze relative performance
            relative_performance = await self._analyze_relative_performance(symbol, sector)

            # Generate market sentiment analysis (pass macro_indicators for prompt inclusion)
            market_sentiment = await self._generate_market_sentiment_analysis(
                symbol, market_context, sector_context, relative_performance, macro_indicators=macro_indicators
            )

            # Compile comprehensive context
            # Defensive check for macro_indicators
            if macro_indicators is None:
                macro_indicators = {}
            context_analysis = {
                "symbol": symbol,
                "sector": sector,
                "market_context": market_context,
                "sector_context": sector_context,
                "relative_performance": relative_performance,
                "market_sentiment": market_sentiment,
                "macro_indicators": macro_indicators.get("macro_summary"),
                "buffett_indicator": macro_indicators.get("buffett_indicator"),
                "macro_fetched_at": macro_indicators.get("fetched_at"),
                "analysis_timestamp": datetime.now().isoformat(),
                "timeframes_analyzed": list(self.timeframes.keys()),
            }

            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"ETF market context analysis completed for {symbol} in {processing_time:.2f}s")

            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result_data=context_analysis,
                processing_time=processing_time,
                metadata={
                    "etfs_analyzed": len(self._get_analyzed_etfs(market_context, sector_context)),
                    "market_regime": market_sentiment.get("market_regime"),
                    "sector_strength": (
                        sector_context.get("sector_strength", "neutral") if sector_context else "unknown"
                    ),
                },
            )

        except Exception as e:
            import traceback

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"ETF market context analysis failed for {symbol}: {e}\n{traceback.format_exc()}")

            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                result_data={},
                processing_time=processing_time,
                error=str(e),
            )

    def _determine_symbol_sector(self, symbol: str, context: Dict) -> Optional[str]:
        """Determine the sector for a given symbol using database lookup"""
        # First check if sector is provided in context
        if "sector" in context:
            sector = context["sector"].lower().replace(" ", "_")
            if sector in self.sector_etfs:
                return sector

        # Query database for sector information
        try:
            query = text('SELECT "Sector" FROM symbol WHERE ticker = :symbol')
            with self.market_data_fetcher.engine.connect() as conn:
                result = conn.execute(query, {"symbol": symbol.upper()}).fetchone()

            if result and result[0]:
                db_sector = result[0].strip()
                # Map database sector names to our ETF sector keys
                sector_mapping = self._map_database_sector_to_etf_key(db_sector)
                if sector_mapping:
                    logger.info(f"Mapped {symbol} sector '{db_sector}' to '{sector_mapping}'")
                    return sector_mapping
                else:
                    logger.warning(f"No ETF mapping found for sector '{db_sector}' for {symbol}")

        except Exception as e:
            logger.warning(f"Failed to lookup sector for {symbol} in database: {e}")

        return None

    def _map_database_sector_to_etf_key(self, db_sector: str) -> Optional[str]:
        """Map database sector names to ETF sector keys"""
        # Mapping from database sector names to our ETF sector keys
        sector_map = {
            "Technology": "technology",
            "Information Technology": "technology",
            "Health Care": "healthcare",
            "Healthcare": "healthcare",
            "Financial Services": "financials",
            "Finance": "financials",
            "Energy": "energy",
            "Industrials": "industrials",
            "Consumer Discretionary": "consumer_discretionary",
            "Consumer Cyclical": "consumer_discretionary",
            "Consumer Staples": "consumer_staples",
            "Consumer Defensive": "consumer_staples",
            "Utilities": "utilities",
            "Basic Materials": "materials",
            "Real Estate": "real_estate",
            "Telecommunications": "communication",
            "Communication Services": "communication",
        }

        return sector_map.get(db_sector)

    async def _analyze_market_context(self, macro_indicators: Dict = None) -> Dict:
        """Analyze overall market context using key ETFs and macro indicators (VIX)"""
        market_context = {}

        for timeframe_name, days in self.timeframes.items():
            timeframe_data = {}

            for market_type, etf_symbol in self.market_etfs.items():
                try:
                    # Get ETF data (exact number of trading days from database)
                    etf_data = self._get_stock_data_cached(etf_symbol, days)

                    if len(etf_data) >= days:
                        # Normalize column names to lowercase for consistency
                        etf_data_normalized = etf_data.copy()
                        etf_data_normalized.columns = [
                            col.lower().replace(" ", "_") for col in etf_data_normalized.columns
                        ]

                        # Calculate returns
                        returns = self._calculate_returns(etf_data_normalized, days)

                        # Calculate volatility
                        volatility = self._calculate_volatility(etf_data_normalized, days)

                        # Calculate momentum indicators
                        momentum = self._calculate_momentum_indicators(etf_data_normalized)

                        timeframe_data[market_type] = {
                            "symbol": etf_symbol,
                            "return": returns["total_return"],
                            "annualized_return": returns["annualized_return"],
                            "volatility": volatility,
                            "momentum": momentum,
                            "current_price": float(etf_data_normalized["close"].iloc[-1]),
                            "days_analyzed": len(etf_data_normalized),
                        }

                    else:
                        logger.warning(f"Insufficient data for {etf_symbol}: {len(etf_data)} days")

                except Exception as e:
                    logger.warning(f"Failed to analyze {market_type} ETF {etf_symbol}: {e}")
                    continue

            market_context[timeframe_name] = timeframe_data

        # Add market regime analysis (now uses VIX from macro indicators)
        market_context["market_regime"] = self._determine_market_regime(market_context, macro_indicators)

        return market_context

    async def _analyze_sector_context(self, sector: str) -> Dict:
        """Analyze sector context using sector ETFs"""
        if sector not in self.sector_etfs:
            return {}

        sector_mapping = self.sector_etfs[sector]
        sector_context = {
            "sector": sector,
            "primary_etf": sector_mapping.primary_etf,
            "description": sector_mapping.description,
        }

        # Analyze primary sector ETF
        all_etfs = [sector_mapping.primary_etf] + sector_mapping.secondary_etfs

        for timeframe_name, days in self.timeframes.items():
            timeframe_data = {}

            for etf_symbol in all_etfs[:3]:  # Analyze up to 3 ETFs per sector
                try:
                    # Get ETF data (exact number of trading days from database)
                    etf_data = self._get_stock_data_cached(etf_symbol, days)

                    if len(etf_data) >= days:
                        # Normalize column names to lowercase for consistency
                        etf_data_normalized = etf_data.copy()
                        etf_data_normalized.columns = [
                            col.lower().replace(" ", "_") for col in etf_data_normalized.columns
                        ]

                        returns = self._calculate_returns(etf_data_normalized, days)
                        volatility = self._calculate_volatility(etf_data_normalized, days)
                        momentum = self._calculate_momentum_indicators(etf_data_normalized)

                        timeframe_data[etf_symbol] = {
                            "return": returns["total_return"],
                            "annualized_return": returns["annualized_return"],
                            "volatility": volatility,
                            "momentum": momentum,
                            "current_price": float(etf_data_normalized["close"].iloc[-1]),
                        }

                except Exception as e:
                    logger.warning(f"Failed to analyze sector ETF {etf_symbol}: {e}")
                    continue

            sector_context[timeframe_name] = timeframe_data

        # Add sector strength vs market
        sector_context["sector_strength"] = self._calculate_sector_strength(sector_context)

        return sector_context

    async def _analyze_relative_performance(self, symbol: str, sector: Optional[str]) -> Dict:
        """Analyze symbol's performance relative to market and sector"""
        relative_performance = {"symbol": symbol}

        try:
            # Get symbol data
            symbol_data = self._get_stock_data_cached(symbol, max(self.timeframes.values()) + 10)

            if len(symbol_data) < 20:
                logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} days")
                return relative_performance

            # Normalize column names for symbol data
            symbol_data.columns = [col.lower().replace(" ", "_") for col in symbol_data.columns]

            # Compare to SPY (market)
            spy_data = self._get_stock_data_cached("SPY", max(self.timeframes.values()) + 10)
            if len(spy_data) > 0:
                spy_data.columns = [col.lower().replace(" ", "_") for col in spy_data.columns]

            if len(spy_data) >= 20:
                market_comparison = {}

                for timeframe_name, days in self.timeframes.items():
                    if len(symbol_data) >= days and len(spy_data) >= days:
                        symbol_return = self._calculate_returns(symbol_data, days)["total_return"]
                        market_return = self._calculate_returns(spy_data, days)["total_return"]

                        market_comparison[timeframe_name] = {
                            "symbol_return": symbol_return,
                            "market_return": market_return,
                            "relative_return": symbol_return - market_return,
                            "beta": self._calculate_beta(symbol_data, spy_data, days),
                        }

                relative_performance["vs_market"] = market_comparison

            # Compare to sector if available
            if sector and sector in self.sector_etfs:
                sector_etf = self.sector_etfs[sector].primary_etf
                sector_data = self._get_stock_data_cached(sector_etf, max(self.timeframes.values()) + 10)

                if len(sector_data) >= 20:
                    # Normalize column names for sector data
                    sector_data.columns = [col.lower().replace(" ", "_") for col in sector_data.columns]
                    sector_comparison = {}

                    for timeframe_name, days in self.timeframes.items():
                        if len(symbol_data) >= days and len(sector_data) >= days:
                            symbol_return = self._calculate_returns(symbol_data, days)["total_return"]
                            sector_return = self._calculate_returns(sector_data, days)["total_return"]

                            sector_comparison[timeframe_name] = {
                                "symbol_return": symbol_return,
                                "sector_return": sector_return,
                                "relative_return": symbol_return - sector_return,
                                "relative_strength": symbol_return / sector_return if sector_return != 0 else 1.0,
                            }

                    relative_performance["vs_sector"] = sector_comparison

        except Exception as e:
            logger.error(f"Failed to calculate relative performance for {symbol}: {e}")

        return relative_performance

    async def _generate_market_sentiment_analysis(
        self,
        symbol: str,
        market_context: Dict,
        sector_context: Dict,
        relative_performance: Dict,
        macro_indicators: Dict = None,
    ) -> Dict:
        """Generate LLM-powered market sentiment analysis with macro economic context"""

        # Defensive checks for None values
        if market_context is None:
            market_context = {}
        if sector_context is None:
            sector_context = {}
        if relative_performance is None:
            relative_performance = {}

        # Prepare data for LLM analysis (including macro economic indicators)
        analysis_data = {
            "market_performance": self._extract_key_market_metrics(market_context),
            "sector_performance": self._extract_key_sector_metrics(sector_context),
            "relative_performance_summary": self._summarize_relative_performance(relative_performance),
            "market_regime": market_context.get("market_regime", "neutral"),
            "macro_indicators": macro_indicators,
        }

        # Build timeframe context string for the prompt
        timeframe_context = []
        for tf_name, tf_days in self.timeframes.items():
            meta = self.timeframe_metadata[tf_name]
            timeframe_context.append(f"  • {meta['label']}: {meta['description']}")

        prompt = f"""
        Analyze the current market and sector context across MULTIPLE TIMEFRAMES with COMPREHENSIVE INSIGHTS:

        TIME HORIZONS ANALYZED (Industry-Standard Trading Days):
{chr(10).join(timeframe_context)}

        Market Data (ALL 4 TIMEFRAMES):
        {self._format_market_data_for_prompt(analysis_data)}

        CRITICAL REQUIREMENT: You MUST analyze ALL FOUR timeframes (10d, 21d, 63d, 252d) for EVERY metric.
        The analysis should identify:
        1. CONFIRMATION - Do all timeframes show the same signal? (Strong conviction)
        2. DIVERGENCE - Are short-term and long-term timeframes contradicting? (Warning sign)
        3. EARLY SIGNALS - Is 10d leading indicator showing trend change before 21d?
        4. TREND STRENGTH - Does 252d baseline support or contradict the shorter-term moves?

        Return structured JSON with this EXACT format:
        {{
          "timeframe_analysis": {{
            "leading_10d": {{"sentiment": "bullish/bearish/neutral", "key_signals": [...]}},
            "short_term_21d": {{"sentiment": "bullish/bearish/neutral", "key_signals": [...]}},
            "medium_term_63d": {{"sentiment": "bullish/bearish/neutral", "key_signals": [...]}},
            "long_term_252d": {{"sentiment": "bullish/bearish/neutral", "key_signals": [...]}}
          }},
          "cross_timeframe_signals": {{
            "confirmation_strength": "strong/moderate/weak/conflicting",
            "divergences": ["list any contradictions between timeframes"],
            "leading_indicators": ["early warning signals from 10d period"],
            "primary_conclusion": "Overall conclusion with PRIMARY focus on 21d, but noting divergences"
          }},
          "overall_sentiment": {{
            "primary_timeframe": "short_term_21d",
            "sentiment": "bullish/bearish/neutral",
            "confidence": "high/medium/low (based on timeframe agreement)",
            "explanation": "Rich explanation with nuanced interpretation (e.g., 'risk_off with selective risk-on in tech')"
          }},
          "risk_environment": {{
            "risk_on_off": "risk_on/risk_off/mixed",
            "timeframe_consensus": "Do all 4 timeframes agree on risk environment?",
            "explanation": "Detailed explanation with nuance (e.g., bonds show risk-off but equities/small-caps risk-on, creating mixed environment with selective risk-taking)"
          }},
          "sector_rotation": {{
            "rotation_trend": "Specific description (e.g., 'Technology leading with strong momentum, small cap strength suggests growth rotation')",
            "timeframe_validation": "Is rotation confirmed across 21d, 63d, 252d?",
            "explanation": "Detailed sector dynamics across all timeframes with investment implications"
          }},
          "key_drivers": [
            "Driver 1 with timeframe support and detailed reasoning (e.g., 'Tech sector strength (AI, cloud, semiconductors) - confirmed 10d/21d/63d +12-15%, weakening 252d due to valuation concerns')",
            "Driver 2 with timeframe support and impact explanation",
            "Driver 3 with timeframe support and market implication"
          ],
          "investment_implications": {{
            "short_term_tactical": "Specific actionable guidance based on 10d/21d signals",
            "medium_term_positioning": "Portfolio positioning based on 63d confirmation",
            "long_term_strategic": "Strategic allocation based on 252d baseline and valuation (Buffett Indicator)",
            "divergence_warnings": ["Specific warnings about conflicting signals"]
          }},
          "market_regime": {{
            "regime": "risk_on/risk_off/transition/mixed (with nuance, e.g., 'risk_off with selective risk-on')",
            "timeframe_stability": "Is regime consistent across timeframes or shifting?",
            "explanation": "Detailed regime analysis explaining contradictions and nuances"
          }},
          "inflation_deflation_signals": {{
            "signal": "inflationary/deflationary/mixed",
            "timeframe_evolution": "How inflation signals are evolving across 10d/21d/63d/252d",
            "explanation": "Analyze gold (inflation hedge), commodities (DBC), oil (demand proxy), CPI data. E.g., 'Gold +19% suggests inflation concerns, but oil -5.8% indicates deflationary pressure from reduced economic activity'"
          }},
          "economic_cycle_position": {{
            "position": "early expansion/mid expansion/late expansion/early contraction/mid contraction/late contraction",
            "timeframe_confirmation": "Is cycle position consistent across timeframes?",
            "explanation": "Use unemployment, GDP, debt levels, market performance to assess cycle position. E.g., 'Strong tech (late expansion signal) but oil decline and risk-off regime suggest early contraction transition'"
          }}
        }}

        EXAMPLES OF GOOD CROSS-TIMEFRAME ANALYSIS:
        - "Tech sector shows +15% (10d), +12% (21d), +8% (63d), +5% (252d) - strong uptrend accelerating driven by AI demand"
        - "SPY +5% (10d/21d) but -2% (252d) with Buffett Indicator 219 - short-term rally within overvalued market (caution)"
        - "Gold +20% (10d) vs +2% (21d/63d/252d), CPI 2.4% - early inflation spike, likely temporary"
        - "Bonds -3% (10d/21d) but +10% (252d), unemployment 4.1% stable - short-term selloff in longer bull market"

        COMPREHENSIVE ANALYSIS REQUIREMENTS - Consider across ALL timeframes:

        MARKET STRUCTURE:
        - Bond vs equity performance (flight to safety or risk-on?)
        - Growth vs value rotation (sector rotation strength)
        - Large cap vs small cap (risk appetite indicator)
        - International vs domestic (global risk sentiment)

        SAFE HAVEN & INFLATION:
        - Precious metals (gold/silver) - safe haven demand + inflation hedge
        - Energy/oil - economic activity proxy + inflation component
        - Commodity complex (DBC) - broad inflation trajectory
        - VIX level - market stress indicator

        ECONOMIC FUNDAMENTALS:
        - Buffett Indicator - market valuation vs GDP (overvalued >1.8, fair 1.0-1.2, undervalued <1.0)
        - Unemployment rate - labor market health
        - GDP growth - economic expansion/contraction
        - CPI - inflation pressure
        - Debt levels (household + public) - systemic risk

        NUANCED INTERPRETATION:
        - Identify contradictions (e.g., "risk-off regime but tech showing risk-on behavior")
        - Explain selective behaviors (e.g., "selective risk-taking in tech despite broader caution")
        - Connect macro to market (e.g., "oil decline -5.8% signals reduced economic activity, confirming risk-off regime")
        - Use specific drivers (e.g., "Tech driven by AI/cloud/semiconductor demand")
        """

        try:
            response = await self.ollama.generate(
                model=self.market_context_model,
                prompt=prompt,
                system="You are a market analyst providing institutional-quality market context analysis.",
                format="json",
                temperature=0.6,
            )

            # Ensure response is a dict
            response_data = response if isinstance(response, dict) else {"analysis": response}

            # DUAL CACHING: First, cache the raw LLM response separately for audit/debugging
            await self._cache_llm_response(
                response=response_data,
                model=self.market_context_model,
                symbol=symbol,  # Symbol from method parameter
                llm_type="market_sentiment_analysis",
                prompt=prompt,
                temperature=0.6,
                top_p=0.9,
                format="json",
            )

            # Then wrap the response for use in agent analysis
            return self._wrap_llm_response(
                response=response_data,
                model=self.market_context_model,
                prompt=prompt,
                temperature=0.6,
                top_p=0.9,
                format="json",
            )

        except Exception as e:
            logger.error(f"Failed to generate market sentiment analysis: {e}")
            return {"error": str(e), "sentiment": "neutral"}

    def _calculate_returns(self, data: pd.DataFrame, days: int) -> Dict:
        """Calculate returns for given period"""
        if len(data) < days:
            return {"total_return": 0, "annualized_return": 0}

        start_price = float(data["close"].iloc[-days])
        end_price = float(data["close"].iloc[-1])

        total_return = (end_price - start_price) / start_price
        annualized_return = (1 + total_return) ** (252 / days) - 1  # 252 trading days per year

        return {"total_return": total_return, "annualized_return": annualized_return}

    def _calculate_volatility(self, data: pd.DataFrame, days: int) -> float:
        """Calculate annualized volatility"""
        if len(data) < days:
            return 0

        recent_data = data.tail(days)
        daily_returns = recent_data["close"].pct_change(fill_method=None).dropna()

        return float(daily_returns.std() * np.sqrt(252))  # Annualized volatility

    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate momentum indicators"""
        if len(data) < 20:
            return {}

        close_prices = data["close"]

        # RSI (14-day)
        rsi = self._calculate_rsi(close_prices, 14)

        # Moving averages
        sma_20 = close_prices.rolling(window=20).mean().iloc[-1] if len(data) >= 20 else close_prices.iloc[-1]
        sma_50 = close_prices.rolling(window=50).mean().iloc[-1] if len(data) >= 50 else close_prices.iloc[-1]

        current_price = close_prices.iloc[-1]

        return {
            "rsi": float(rsi) if not pd.isna(rsi) else 50,
            "price_vs_sma20": float((current_price - sma_20) / sma_20) if sma_20 > 0 else 0,
            "price_vs_sma50": float((current_price - sma_50) / sma_50) if sma_50 > 0 else 0,
            "sma20_vs_sma50": float((sma_20 - sma_50) / sma_50) if sma_50 > 0 else 0,
        }

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if len(rsi) > 0 else 50

    def _calculate_beta(self, symbol_data: pd.DataFrame, market_data: pd.DataFrame, days: int) -> float:
        """Calculate beta vs market"""
        if len(symbol_data) < days or len(market_data) < days:
            return 1.0

        # Align data by date
        symbol_recent = symbol_data.tail(days)["close"].pct_change(fill_method=None).dropna()
        market_recent = market_data.tail(days)["close"].pct_change(fill_method=None).dropna()

        # Ensure same length
        min_length = min(len(symbol_recent), len(market_recent))
        if min_length < 10:
            return 1.0

        symbol_returns = symbol_recent.tail(min_length)
        market_returns = market_recent.tail(min_length)

        covariance = np.cov(symbol_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)

        return covariance / market_variance if market_variance > 0 else 1.0

    def _determine_market_regime(self, market_context: Dict, macro_indicators: Dict = None) -> str:
        """Determine current market regime using multi-factor analysis including commodities and VIX"""
        # Analyze medium-term performance
        medium_term = market_context.get("medium_term", {})

        if not medium_term:
            return "neutral"

        # Core market data
        spy_data = medium_term.get("broad_market", {})
        bond_data = medium_term.get("bonds", {})
        small_cap_data = medium_term.get("small_cap", {})

        # Commodity data for inflation/cycle detection
        gold_data = medium_term.get("gold", {})
        silver_data = medium_term.get("silver", {})
        oil_data = medium_term.get("oil", {})
        commodity_data = medium_term.get("commodities", {})

        spy_return = spy_data.get("return", 0)
        bond_return = bond_data.get("return", 0)
        small_cap_return = small_cap_data.get("return", 0)
        gold_return = gold_data.get("return", 0)
        oil_return = oil_data.get("return", 0)
        commodity_return = commodity_data.get("return", 0)

        # Enhanced regime classification using multiple factors
        risk_signals = 0
        total_signals = 0

        # Signal 1: Equity vs Bond performance
        if spy_return > bond_return:
            risk_signals += 1
        total_signals += 1

        # Signal 2: Small cap vs Large cap (risk appetite)
        if small_cap_return > spy_return:
            risk_signals += 1
        total_signals += 1

        # Signal 3: VIX levels from macro indicators (VIXCLS from FRED database)
        if macro_indicators:
            macro_summary = macro_indicators.get("macro_summary", {})
            if macro_summary and "VIXCLS" in macro_summary:
                vix_level = macro_summary["VIXCLS"].get("value")
                if vix_level:
                    if vix_level < 20:  # Low volatility = risk-on
                        risk_signals += 1
                    elif vix_level > 30:  # High volatility = risk-off
                        risk_signals -= 1
                    total_signals += 1

        # Signal 4: Absolute equity performance
        if spy_return > 0.05:  # Strong positive returns
            risk_signals += 1
        elif spy_return < -0.05:  # Strong negative returns
            risk_signals -= 1
        total_signals += 1

        # Signal 5: Precious metals (safe haven demand)
        if gold_return < -0.02:  # Gold declining = less safe haven demand = risk-on
            risk_signals += 1
        elif gold_return > 0.05:  # Gold rallying = safe haven demand = risk-off
            risk_signals -= 1
        total_signals += 1

        # Signal 6: Oil/Energy (economic activity indicator)
        if oil_return > 0.03:  # Oil rising = economic growth = risk-on
            risk_signals += 1
        elif oil_return < -0.05:  # Oil falling = economic slowdown = risk-off
            risk_signals -= 1
        total_signals += 1

        # Signal 7: Broad commodities (inflation/growth cycle)
        if commodity_return > 0.03:  # Commodities rising = growth/inflation = risk-on
            risk_signals += 1
        elif commodity_return < -0.05:  # Commodities falling = deflationary = risk-off
            risk_signals -= 1
        total_signals += 1

        # Calculate risk-on ratio
        risk_ratio = risk_signals / max(total_signals, 1)

        # Classify regime with enhanced thresholds
        if risk_ratio >= 0.65:  # Slightly higher threshold with more signals
            return "risk_on"
        elif risk_ratio <= 0.35:  # Slightly lower threshold with more signals
            return "risk_off"
        else:
            return "mixed"

    def _calculate_sector_strength(self, sector_context: Dict) -> str:
        """Calculate sector strength vs market"""
        medium_term = sector_context.get("medium_term", {})

        if not medium_term:
            return "neutral"

        # Get primary ETF performance
        primary_etf = sector_context.get("primary_etf")
        etf_data = medium_term.get(primary_etf, {})

        etf_return = etf_data.get("return", 0)

        if etf_return > 0.05:
            return "strong"
        elif etf_return > 0:
            return "moderate"
        elif etf_return > -0.05:
            return "weak"
        else:
            return "very_weak"

    def _get_analyzed_etfs(self, market_context: Dict, sector_context: Dict) -> List[str]:
        """Get list of all ETFs analyzed"""
        etfs = set()

        # Add market ETFs
        for timeframe_data in market_context.values():
            if isinstance(timeframe_data, dict):
                for market_data in timeframe_data.values():
                    if isinstance(market_data, dict) and "symbol" in market_data:
                        etfs.add(market_data["symbol"])

        # Add sector ETFs
        for timeframe_data in sector_context.values():
            if isinstance(timeframe_data, dict):
                for etf_symbol in timeframe_data.keys():
                    if etf_symbol not in ["sector", "primary_etf", "description", "sector_strength"]:
                        etfs.add(etf_symbol)

        return list(etfs)

    def _extract_key_market_metrics(self, market_context: Dict) -> Dict:
        """Extract key metrics for LLM analysis including commodities"""
        medium_term = market_context.get("medium_term", {})

        return {
            "spy_return": medium_term.get("broad_market", {}).get("return", 0),
            "bonds_return": medium_term.get("bonds", {}).get("return", 0),
            "small_cap_return": medium_term.get("small_cap", {}).get("return", 0),
            "international_return": medium_term.get("international", {}).get("return", 0),
            "gold_return": medium_term.get("gold", {}).get("return", 0),
            "silver_return": medium_term.get("silver", {}).get("return", 0),
            "oil_return": medium_term.get("oil", {}).get("return", 0),
            "commodities_return": medium_term.get("commodities", {}).get("return", 0),
            "market_regime": market_context.get("market_regime", "neutral"),
        }

    def _extract_key_sector_metrics(self, sector_context: Dict) -> Dict:
        """Extract key sector metrics"""
        if not sector_context:
            return {}

        medium_term = sector_context.get("medium_term", {})
        primary_etf = sector_context.get("primary_etf")

        primary_data = medium_term.get(primary_etf, {}) if primary_etf else {}

        return {
            "sector": sector_context.get("sector"),
            "sector_return": primary_data.get("return", 0),
            "sector_strength": sector_context.get("sector_strength", "neutral"),
        }

    def _summarize_relative_performance(self, relative_performance: Dict) -> Dict:
        """Summarize relative performance data"""
        summary = {}

        vs_market = relative_performance.get("vs_market", {})
        vs_sector = relative_performance.get("vs_sector", {})

        if vs_market:
            medium_term = vs_market.get("medium_term", {})
            summary["vs_market"] = {
                "relative_return": medium_term.get("relative_return", 0),
                "beta": medium_term.get("beta", 1.0),
            }

        if vs_sector:
            medium_term = vs_sector.get("medium_term", {})
            summary["vs_sector"] = {
                "relative_return": medium_term.get("relative_return", 0),
                "relative_strength": medium_term.get("relative_strength", 1.0),
            }

        return summary

    def _format_market_data_for_prompt(self, analysis_data: Dict) -> str:
        """Format market data for LLM prompt including commodities and macro indicators"""
        lines = []

        market_perf = analysis_data.get("market_performance", {})
        lines.append(f"Market Performance (1M):")
        lines.append(f"  SPY: {market_perf.get('spy_return', 0):.2%}")
        lines.append(f"  Bonds: {market_perf.get('bonds_return', 0):.2%}")
        lines.append(f"  Small Cap: {market_perf.get('small_cap_return', 0):.2%}")
        lines.append(f"  International: {market_perf.get('international_return', 0):.2%}")
        lines.append(f"  Market Regime: {market_perf.get('market_regime', 'neutral')}")

        # Add commodity performance section
        lines.append(f"\nCommodity Performance (1M):")
        lines.append(f"  Gold (GLD): {market_perf.get('gold_return', 0):.2%}")
        lines.append(f"  Silver (SLV): {market_perf.get('silver_return', 0):.2%}")
        lines.append(f"  Oil (USO): {market_perf.get('oil_return', 0):.2%}")
        lines.append(f"  Commodities (DBC): {market_perf.get('commodities_return', 0):.2%}")

        sector_perf = analysis_data.get("sector_performance", {})
        if sector_perf:
            lines.append(f"\nSector Performance:")
            lines.append(f"  Sector: {sector_perf.get('sector', 'unknown')}")
            lines.append(f"  Return: {sector_perf.get('sector_return', 0):.2%}")
            lines.append(f"  Strength: {sector_perf.get('sector_strength', 'neutral')}")

        # Add macro economic indicators section
        macro_indicators = analysis_data.get("macro_indicators")
        if macro_indicators:
            lines.append(f"\nMacro Economic Indicators:")

            # Buffett Indicator (Market Cap / GDP ratio) - token-efficient rounding
            buffett = macro_indicators.get("buffett_indicator")
            if buffett:
                lines.append(
                    f"  Buffett Indicator: {buffett.get('ratio', 0):.1f}% ({buffett.get('interpretation', 'N/A')})"
                )

            # Extract macro_summary which contains FRED data - apply judicious rounding
            macro_summary = macro_indicators.get("macro_summary") or {}

            # GDP per capita (whole numbers sufficient)
            gdp_per_capita = macro_summary.get("GDP_PER_CAPITA")
            if gdp_per_capita:
                lines.append(f"  GDP/Capita: ${gdp_per_capita.get('value', 0):,.0f}")

            # CPI (1 decimal sufficient)
            cpi = macro_summary.get("CPIAUCSL")
            if cpi:
                lines.append(f"  CPI: {cpi.get('value', 0):.1f}")
                change = cpi.get("yoy_change")
                if change:
                    lines.append(f"    YoY: {change:.1%}")

            # Household Debt to GDP (1 decimal)
            household_debt = macro_summary.get("HDTGPDUSQ163N")
            if household_debt:
                lines.append(f"  Household Debt/GDP: {household_debt.get('value', 0):.1f}%")

            # Public Debt to GDP (1 decimal)
            public_debt = macro_summary.get("GFDEGDQ188S")
            if public_debt:
                lines.append(f"  Public Debt/GDP: {public_debt.get('value', 0):.1f}%")

            # Unemployment Rate (1 decimal)
            unemployment = macro_summary.get("UNRATE")
            if unemployment:
                lines.append(f"  Unemployment: {unemployment.get('value', 0):.1f}%")

            # VIX (1 decimal)
            vix = macro_summary.get("VIXCLS")
            if vix:
                lines.append(f"  VIX: {vix.get('value', 0):.1f}")

            # Add timestamp of macro data fetch
            fetched_at = macro_indicators.get("fetched_at")
            if fetched_at:
                lines.append(f"  Data Fetched: {fetched_at}")

        return "\n".join(lines)
