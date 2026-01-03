"""
Technical Analysis Agent
Specialized agent for technical analysis and market data processing using Ollama LLMs
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from investigator.domain.agents.base import InvestmentAgent
from investigator.domain.models.analysis import AgentResult, AgentTask, TaskStatus
from investigator.infrastructure.cache import CacheManager
from investigator.infrastructure.cache.cache_types import CacheType
from investigator.domain.services.toon_formatter import to_toon_array, TOONFormatter
from investigator.infrastructure.database.market_data import get_market_data_fetcher  # Uses singleton pattern
from investigator.infrastructure.indicators import TechnicalIndicatorCalculator as TechnicalIndicators


class TechnicalAnalysisAgent(InvestmentAgent):
    """
    Agent specialized in technical analysis and price action patterns

    Pipeline cheat-sheet:

        fetch OHLCV ──▶ compute indicators ──▶ prompt LLM ──▶ cache outputs
              ▲                │                     │
              │                └── also stores parquet snapshots on hit
              └──── caches reuse DataFrames for future runs
    """

    def __init__(self, agent_id: str, ollama_client, event_bus, cache_manager: CacheManager):
        # Load config for MarketDataFetcher
        from investigator.config import get_config

        config = get_config()
        self.config = config  # Store config for TOON format access
        self.technical_model = config.ollama.models.get("technical_analysis", "deepseek-r1:32b")

        # Specialized models for different analysis types (must exist before super().__init__)
        self.models = {
            "pattern_recognition": self.technical_model,
            "trend_analysis": self.technical_model,
            "signal_generation": self.technical_model,
        }

        super().__init__(agent_id, ollama_client, event_bus, cache_manager)

        self.market_data = get_market_data_fetcher(config)
        self.indicators = TechnicalIndicators()

        # Technical analysis parameters
        self.timeframes = {"intraday": "1d", "short": "1mo", "medium": "3mo", "long": "1y", "extended": "5y"}

        # Key technical patterns to detect
        self.patterns = [
            "head_and_shoulders",
            "inverse_head_and_shoulders",
            "double_top",
            "double_bottom",
            "ascending_triangle",
            "descending_triangle",
            "flag",
            "pennant",
            "wedge",
            "cup_and_handle",
            "rounding_bottom",
        ]

    def _debug_log_prompt(self, label: str, prompt: str) -> None:
        if self.logger.isEnabledFor(logging.DEBUG):
            trimmed = prompt if len(prompt) <= 6000 else f"{prompt[:6000]}\n...[truncated]"
            self.logger.debug("📤 %s PROMPT:\n%s", label, trimmed)

    def _debug_log_response(self, label: str, response: Any) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return

        if isinstance(response, (dict, list)):
            try:
                payload = json.dumps(response, indent=2, default=str)
            except (TypeError, ValueError):
                payload = str(response)
        else:
            payload = str(response)

        if len(payload) > 6000:
            payload = f"{payload[:6000]}\n...[truncated]"

        self.logger.debug("📥 %s RESPONSE:\n%s", label, payload)

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
                analysis_type=AnalysisType.TECHNICAL_ANALYSIS,
                min_data_required={"symbol": str, "period": str},
                max_processing_time=240,  # Increased 2x for slower hardware
                required_models=[self.technical_model],
                cache_ttl=1800,
            )
        ]

    async def process(self, task: AgentTask) -> AgentResult:
        """Process technical analysis task"""
        symbol = task.context.get("symbol")
        timeframe = task.context.get("timeframe", "medium")
        analysis_type = task.context.get("analysis_type", "comprehensive")

        self.logger.info(f"Performing {analysis_type} technical analysis for {symbol}")

        try:
            # Fetch market data
            price_data = await self._fetch_price_data(symbol, timeframe)

            # Calculate technical indicators
            indicators = await self._calculate_indicators(price_data, symbol)

            # Detect chart patterns
            patterns = await self._detect_patterns(price_data, symbol)

            # Analyze volume patterns
            volume_analysis = await self._analyze_volume(price_data)

            # Generate support/resistance levels
            levels = await self._identify_key_levels(price_data)

            # Analyze momentum and trend
            momentum = await self._analyze_momentum(indicators)

            # Generate trading signals
            signals = await self._generate_signals(price_data, indicators, patterns, momentum, symbol)

            # Market sentiment analysis
            sentiment = await self._analyze_market_sentiment(price_data, volume_analysis)

            # Handle both uppercase and lowercase columns for current_price
            close_col = "close" if "close" in price_data.columns else "Close"

            # Synthesize technical report
            report = await self._synthesize_technical_report(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "current_price": price_data[close_col].iloc[-1],
                    "price_change": self._calculate_price_change(price_data),
                    "indicators": indicators,
                    "patterns": patterns,
                    "volume_analysis": volume_analysis,
                    "support_resistance": levels,
                    "momentum": momentum,
                    "signals": signals,
                    "sentiment": sentiment,
                }
            )

            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result_data={
                    "status": "success",
                    "symbol": symbol,
                    "analysis": report,
                    "signals": signals,
                    "levels": levels,
                    "technical_rating": report.get("technical_rating", 0),
                    "recommendation": report.get("recommendation", "neutral"),
                },
                processing_time=0,  # Will be calculated by base class
            )

        except Exception as e:
            self.logger.error(f"Technical analysis failed for {symbol}: {e}")
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                result_data={"status": "error", "symbol": symbol, "error": str(e)},
                processing_time=0,
                error=str(e),
            )

    async def _fetch_price_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch historical price data"""
        cache_key = f"price_data:{symbol}:{timeframe}"

        # Check cache (1 hour TTL for price data)
        cache_key = {"symbol": symbol, "timeframe": timeframe}
        cached = self.cache.get(CacheType.TECHNICAL_DATA, cache_key) if self.cache else None
        if cached is not None:
            if isinstance(cached, pd.DataFrame):
                if not cached.empty:
                    return cached
            if isinstance(cached, dict):
                cached_df = None
                if isinstance(cached.get("dataframe"), pd.DataFrame):
                    cached_df = cached["dataframe"]
                elif "data" in cached:
                    cached_df = pd.DataFrame(cached["data"])
                if cached_df is not None and not cached_df.empty:
                    return cached_df
            if isinstance(cached, (list, tuple)) and cached:
                cached_df = pd.DataFrame(cached)
                if not cached_df.empty:
                    return cached_df
            self.logger.warning("Cached technical data for %s/%s was empty; refreshing", symbol, timeframe)

        # Fetch from market data provider
        period = self.timeframes.get(timeframe, "3mo")
        data = await self.market_data.get_historical_data(symbol, period)

        if data is None or data.empty:
            fallback = Path("data") / "technical_cache" / symbol.upper() / f"technical_data_{symbol.upper()}.csv"
            if fallback.exists():
                self.logger.warning(
                    "Primary market data fetch returned no rows for %s; " "loading fallback dataset %s",
                    symbol,
                    fallback,
                )
                data = pd.read_csv(fallback, parse_dates=["Date"], index_col="Date")
                for col in data.columns:
                    if data[col].dtype == object:
                        cleaned = data[col].astype(str).str.replace(",", "")
                        numeric = pd.to_numeric(cleaned, errors="coerce")
                        # Preserve original text columns (e.g., pattern labels)
                        if numeric.notna().any():
                            data[col] = numeric
            else:
                raise ValueError(f"No market data available for {symbol}")

        # Cache the data
        if self.cache and not data.empty:
            try:
                self.cache.set(CacheType.TECHNICAL_DATA, cache_key, data)
            except Exception as e:
                self.logger.warning(f"Failed to cache technical data: {e}")

        return data

    async def _calculate_indicators(self, price_data: pd.DataFrame, symbol: str = "unknown") -> Dict:
        """Calculate comprehensive technical indicators"""
        # Standardize column names to match expected format (lowercase)
        price_data_standardized = price_data.copy()
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "AdjClose": "adjclose",  # Without space
            "Adj Close": "adjclose",  # With space (from market data fetcher)
        }
        price_data_standardized = price_data_standardized.rename(columns=column_mapping)

        # Use the centralized indicator calculator
        enhanced_df = self.indicators.calculate_all_indicators(price_data_standardized, symbol)

        # Extract the latest indicator values as a dict
        latest_values = enhanced_df.iloc[-1].to_dict()

        # Clean up column names and organize the data
        indicators = {}

        # Moving averages
        indicators["sma_20"] = latest_values.get("SMA_20", 0)
        indicators["sma_50"] = latest_values.get("SMA_50", 0)
        indicators["sma_200"] = latest_values.get("SMA_200", 0)
        indicators["ema_12"] = latest_values.get("EMA_12", 0)
        indicators["ema_26"] = latest_values.get("EMA_26", 0)

        # MACD components
        indicators["macd"] = latest_values.get("MACD", 0)
        indicators["macd_signal"] = latest_values.get("MACD_Signal", 0)
        indicators["macd_histogram"] = latest_values.get("MACD_Histogram", 0)

        # RSI
        indicators["rsi"] = latest_values.get("RSI_14", 50)

        # Bollinger Bands
        indicators["bb_upper"] = latest_values.get("BB_Upper", 0)
        indicators["bb_middle"] = latest_values.get("BB_Middle", 0)
        indicators["bb_lower"] = latest_values.get("BB_Lower", 0)

        # Stochastic
        indicators["stoch_k"] = latest_values.get("Stoch_K", 50)
        indicators["stoch_d"] = latest_values.get("Stoch_D", 50)

        # ATR
        indicators["atr"] = latest_values.get("ATR", 0)

        # Volume indicators
        indicators["obv"] = latest_values.get("OBV", 0)
        indicators["vwap"] = latest_values.get("VWAP", latest_values.get("Close", 0))

        return indicators

    async def _detect_patterns(self, price_data: pd.DataFrame, symbol: str) -> List[Dict]:
        """Detect chart patterns using pattern recognition"""
        detected_patterns = []

        # Handle both uppercase and lowercase columns
        columns_to_use = []
        for col in ["open", "high", "low", "close", "volume"]:
            if col in price_data.columns:
                columns_to_use.append(col)
            elif col.capitalize() in price_data.columns:
                columns_to_use.append(col.capitalize())

        if len(columns_to_use) < 5:
            # Not enough data columns for pattern analysis
            return []

        # Prepare data for pattern analysis
        price_array = np.round(price_data[columns_to_use].values, 2)

        prompt = f"""
        Analyze the following daily price data for technical chart patterns. Infer both daily (short-to-medium term) and weekly (long-term) patterns from this data.
        
        Recent 252 periods of OHLCV data:
        {price_array[-252:].tolist()}
        
        Detect the following patterns if present:
        - Reversal Patterns: Head and Shoulders (regular and inverse), Double/Triple Top/Bottom, Rounding Bottom/Top
        - Continuation Patterns: Triangle (ascending, descending, symmetrical), Flag, Pennant, Wedge (rising, falling), Rectangle
        - Candlestick Patterns: Doji, Hammer, Hanging Man, Engulfing (Bullish/Bearish), Morning/Evening Star
                - Other Patterns: Price Channels, Gaps (Breakaway, Runaway, Exhaustion)
        
                Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.
                
                For each detected pattern, provide:
                - pattern_name
                - confidence (0-100)
                - start_index
                - end_index
                - breakout_level
                - target_price
                - pattern_description
        
                Return a JSON array of pattern objects that follows the schema below (values are illustrative):
        [
          {{
            "pattern_name": "head_and_shoulders",
            "confidence": 85,
            "start_index": 50,
            "end_index": 100,
            "breakout_level": 145.0,
            "target_price": 130.0,
            "pattern_description": "A head and shoulders pattern has formed, indicating a potential trend reversal."
          }}
        ]
        """

        prompt_name = "_detect_patterns_prompt"
        self._debug_log_prompt(prompt_name, prompt)

        response = await self.ollama.generate(
            model=self.models["pattern_recognition"],
            prompt=prompt,
            system="You are a technical analysis expert specializing in chart pattern recognition.",
            format="json",
            prompt_name=prompt_name,
        )

        self._debug_log_response(prompt_name, response)

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["pattern_recognition"],
            symbol=symbol,
            llm_type="technical_pattern_recognition",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Parse response safely
        response_dict = self._parse_llm_response(response)
        patterns = response_dict.get("patterns", [])

        # Validate and enhance patterns with calculations
        for pattern in patterns:
            pattern["validation"] = self._validate_pattern(pattern, price_data)
            detected_patterns.append(pattern)

        # Return the list directly (not wrapped) to avoid slice errors
        # The wrapping is already handled by _cache_llm_response above
        return detected_patterns

    async def _analyze_volume(self, price_data: pd.DataFrame) -> Dict:
        """Analyze volume patterns and anomalies"""
        volume_analysis = {}

        # Handle both uppercase and lowercase columns
        volume_col = "volume" if "volume" in price_data.columns else "Volume"
        if volume_col not in price_data.columns:
            return {"error": "No volume data available"}

        # Calculate volume metrics
        avg_volume = price_data[volume_col].rolling(window=20).mean()
        volume_ratio = price_data[volume_col] / avg_volume

        # Detect volume spikes
        volume_spikes = price_data[volume_ratio > 2.0]

        # Analyze volume trends
        volume_trend = (
            "increasing"
            if price_data[volume_col].iloc[-20:].mean() > price_data[volume_col].iloc[-40:-20].mean()
            else "decreasing"
        )

        # Price-volume correlation
        close_col = "close" if "close" in price_data.columns else "Close"
        price_volume_corr = (
            price_data[close_col].pct_change(fill_method=None).corr(price_data[volume_col].pct_change(fill_method=None))
        )

        volume_analysis = {
            "current_volume": int(price_data[volume_col].iloc[-1]),
            "avg_volume_20d": int(avg_volume.iloc[-1]),
            "volume_trend": volume_trend,
            "recent_spikes": len(volume_spikes[-20:]),
            "price_volume_correlation": float(price_volume_corr),
            "volume_profile": self._calculate_volume_profile(price_data),
            "accumulation_distribution": self._calculate_accumulation_distribution(price_data),
        }

        return volume_analysis

    async def _identify_key_levels(self, price_data: pd.DataFrame) -> Dict:
        """Identify support and resistance levels"""
        levels = {}

        # Handle both uppercase and lowercase columns
        high_col = "high" if "high" in price_data.columns else "High"
        low_col = "low" if "low" in price_data.columns else "Low"
        close_col = "close" if "close" in price_data.columns else "Close"

        # Calculate pivot points
        high = price_data[high_col].iloc[-1]
        low = price_data[low_col].iloc[-1]
        close = price_data[close_col].iloc[-1]

        pivot = (high + low + close) / 3

        levels["pivot_point"] = pivot
        levels["resistance_1"] = 2 * pivot - low
        levels["resistance_2"] = pivot + (high - low)
        levels["resistance_3"] = high + 2 * (pivot - low)
        levels["support_1"] = 2 * pivot - high
        levels["support_2"] = pivot - (high - low)
        levels["support_3"] = low - 2 * (high - pivot)

        # Identify historical support/resistance
        historical_levels = self._find_historical_levels(price_data)
        levels["historical_support"] = historical_levels["support"]
        levels["historical_resistance"] = historical_levels["resistance"]

        # Fibonacci retracements
        levels["fibonacci"] = self._calculate_fibonacci_levels(price_data)

        return levels

    async def _analyze_momentum(self, indicators: Dict) -> Dict:
        """Analyze momentum indicators for trend strength"""
        momentum = {}

        # RSI analysis
        current_rsi = indicators.get("rsi", 50)  # These are already scalar values
        momentum["rsi_signal"] = "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral"

        # MACD analysis
        if "macd" in indicators and "macd_signal" in indicators:
            macd_cross = indicators["macd"] - indicators["macd_signal"]
            momentum["macd_signal"] = "bullish" if macd_cross > 0 else "bearish"
            momentum["macd_strength"] = abs(macd_cross)

        # Stochastic analysis
        if "stoch_k" in indicators:
            stoch_k = indicators["stoch_k"]
            momentum["stochastic_signal"] = "overbought" if stoch_k > 80 else "oversold" if stoch_k < 20 else "neutral"

        # Moving average analysis
        if "sma_50" in indicators and "sma_200" in indicators:
            golden_cross = indicators["sma_50"] > indicators["sma_200"]
            momentum["ma_signal"] = "golden_cross" if golden_cross else "death_cross"

        # Overall momentum score (-100 to 100)
        momentum["overall_score"] = self._calculate_momentum_score(indicators)

        return momentum

    async def _generate_signals(
        self, price_data: pd.DataFrame, indicators: Dict, patterns: List[Dict], momentum: Dict, symbol: str
    ) -> Dict:
        """Generate trading signals based on technical analysis"""
        # Handle both uppercase and lowercase columns
        close_col = "close" if "close" in price_data.columns else "Close"

        prompt = f"""
        Generate trading signals based on the following technical analysis:
        
        Current Price: {price_data[close_col].iloc[-1]}
        Price Change (5 days): {((price_data[close_col].iloc[-1] / price_data[close_col].iloc[-5]) - 1) * 100:.2f}%
        
        Key Indicators:
        - RSI: {indicators.get('rsi', 50):.2f}
        - MACD Signal: {momentum.get('macd_signal', 'neutral')}
        - Moving Average Signal: {momentum.get('ma_signal', 'neutral')}
        - Momentum Score: {momentum.get('overall_score', 0)}
        
        Detected Patterns:
        {json.dumps(patterns[:3], indent=2) if patterns else 'None detected'}
        
        Generate the following signals:
        1. entry_signal (buy/sell/hold)
        2. entry_price
        3. stop_loss
        4. take_profit_1
        5. take_profit_2
        6. position_size_recommendation (conservative/moderate/aggressive)
        7. confidence_level (0-100)
        8. time_horizon (short/medium/long)
        9. risk_reward_ratio
        10. signal_strength (weak/moderate/strong)
        
        Return only a single JSON object that strictly matches the schema below (values are illustrative). Do NOT include any additional narration outside the JSON structure.

        Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.

        Example structure:

        {{
          "entry_signal": "buy",
          "entry_price": 150.0,
          "stop_loss": 145.0,
          "take_profit_1": 160.0,
          "take_profit_2": 170.0,
          "position_size_recommendation": "moderate",
          "confidence_level": 85,
          "time_horizon": "medium",
          "risk_reward_ratio": 2.0,
          "signal_strength": "strong",
          "reasoning": "The stock is showing strong bullish momentum with a golden cross and a breakout from a bullish flag pattern."
        }}
        """

        prompt_name = "_generate_signals_prompt"
        self._debug_log_prompt(prompt_name, prompt)

        response = await self.ollama.generate(
            model=self.models["signal_generation"],
            prompt=prompt,
            format="json",
            prompt_name=prompt_name,
        )

        self._debug_log_response(prompt_name, response)

        # DUAL CACHING: Cache LLM response separately
        await self._cache_llm_response(
            response=response,
            model=self.models["signal_generation"],
            symbol=symbol,
            llm_type="technical_signal_generation",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Wrap with metadata for caching
        return self._wrap_llm_response(
            response=response,
            model=self.models["signal_generation"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

    async def _analyze_market_sentiment(self, price_data: pd.DataFrame, volume_analysis: Dict) -> Dict:
        """Analyze overall market sentiment"""
        sentiment = {}

        # Handle both uppercase and lowercase columns
        close_col = "close" if "close" in price_data.columns else "Close"

        # Price action sentiment
        recent_trend = "bullish" if price_data[close_col].iloc[-1] > price_data[close_col].iloc[-20] else "bearish"

        # Volatility analysis
        returns = price_data[close_col].pct_change(fill_method=None)
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Volume sentiment
        volume_sentiment = "accumulation" if volume_analysis.get("accumulation_distribution", 0) > 0 else "distribution"

        sentiment = {
            "price_trend": recent_trend,
            "volatility": float(volatility),
            "volume_sentiment": volume_sentiment,
            "market_phase": self._identify_market_phase(price_data),
            "trend_strength": self._calculate_trend_strength(price_data),
            "sentiment_score": self._calculate_sentiment_score(price_data, volume_analysis),
        }

        return sentiment

    async def _synthesize_technical_report(self, analysis_data: Dict) -> Dict:
        """Synthesize comprehensive technical analysis report"""
        # Round all numeric values to reduce token usage and improve readability
        from investigator.domain.services.data_normalizer import DataNormalizer

        rounded_data = DataNormalizer.round_financial_data(analysis_data)

        # Check if TOON format is enabled
        use_toon = getattr(self.config.ollama, "use_toon_format", False) and getattr(
            self.config.ollama, "toon_agents", {}
        ).get("technical_analysis", False)

        if use_toon:
            # Convert tabular data to TOON format for 87% token savings
            data_sections = []

            # Extract and convert price data if available
            if "daily_data" in rounded_data:
                try:
                    daily_list = rounded_data["daily_data"]
                    if isinstance(daily_list, list) and len(daily_list) > 0:
                        toon_daily = to_toon_array(daily_list, name="daily_price_data")
                        data_sections.append(toon_daily)
                except Exception as e:
                    self.logger.warning(f"Failed to convert daily_data to TOON: {e}")

            # Extract and convert weekly data if available
            if "weekly_data" in rounded_data:
                try:
                    weekly_list = rounded_data["weekly_data"]
                    if isinstance(weekly_list, list) and len(weekly_list) > 0:
                        toon_weekly = to_toon_array(weekly_list, name="weekly_price_data")
                        data_sections.append(toon_weekly)
                except Exception as e:
                    self.logger.warning(f"Failed to convert weekly_data to TOON: {e}")

            # If we have TOON data, use it; otherwise fall back to JSON
            if data_sections:
                data_section = "\n\n".join(data_sections)

                # Add remaining non-tabular data as JSON
                non_tabular = {k: v for k, v in rounded_data.items() if k not in ["daily_data", "weekly_data"]}
                if non_tabular:
                    data_section += f"\n\nAdditional context:\n{json.dumps(non_tabular, indent=2)}"
            else:
                # No tabular data found, fall back to JSON
                data_section = json.dumps(rounded_data, indent=2)[:8000]
        else:
            # TOON disabled, use JSON (current behavior)
            data_section = json.dumps(rounded_data, indent=2)[:8000]

        prompt = f"""
        Synthesize a comprehensive technical analysis report, considering both daily and weekly perspectives:

        {data_section}
        
        Create a structured technical report with:
        1. Executive Summary (key technical insights)
        2. Trend Analysis (primary, secondary, and long-term weekly trends)
        3. Key Support and Resistance Levels
        4. Technical Indicators Summary
        5. Pattern Recognition Results (comment on both daily and weekly patterns)
        6. Volume Analysis Insights
        7. Momentum and Strength Assessment
        8. Risk Assessment (technical risks)
        9. Trading Recommendations
        10. Technical Rating (1-10)
        11. Time Horizon Recommendations
        12. Key Watch Levels
        
        Be objective and highlight both bullish and bearish scenarios.

        Before generating the JSON, think step-by-step about the analysis. Put your thinking process inside <think> and </think> tags.

        Return a JSON object that follows the schema below (values are illustrative):
        {{
          "executive_summary": "The stock is in a strong uptrend, with bullish momentum and a series of higher highs and higher lows. The technical indicators are bullish, and the stock is trading above its key moving averages.",
          "trend_analysis": {{
            "primary_trend": "Uptrend",
            "secondary_trend": "Uptrend"
          }},
          "key_support_and_resistance_levels": {{
            "support": [140.0, 135.0],
            "resistance": [155.0, 160.0]
          }},
          "technical_indicators_summary": {{
            "rsi": "Bullish (65)",
            "macd": "Bullish (golden cross)",
            "moving_averages": "Bullish (trading above 50-day and 200-day SMA)"
          }},
          "pattern_recognition_results": [
            {{
              "pattern_name": "bullish_flag",
              "confidence": 90
            }}
          ],
          "volume_analysis_insights": "Volume has been increasing on up days, which confirms the strength of the uptrend.",
          "momentum_and_strength_assessment": "Momentum is strong and the trend is well-supported.",
          "risk_assessment": "The main technical risk is a potential pullback to the 50-day moving average.",
          "trading_recommendations": "Buy on dips, with a stop-loss below the 50-day moving average.",
          "technical_rating": 8,
          "time_horizon_recommendations": "Medium-term (1-3 months)",
          "key_watch_levels": {{
            "support": 140.0,
            "resistance": 155.0
          }}
        }}
        """

        prompt_name = "_synthesize_technical_report_prompt"
        self._debug_log_prompt(prompt_name, prompt)

        # Build system prompt with optional TOON explanation
        system_prompt = "You are a senior technical analyst providing comprehensive market analysis."
        if use_toon and data_sections:
            system_prompt += "\n\n" + TOONFormatter.get_format_explanation()

        response = await self.ollama.generate(
            model=self.models["trend_analysis"],
            prompt=prompt,
            system=system_prompt,
            format="json",
            prompt_name=prompt_name,
        )

        self._debug_log_response(prompt_name, response)

        # DUAL CACHING: Cache LLM response separately
        symbol = analysis_data.get("symbol", "UNKNOWN")
        await self._cache_llm_response(
            response=response,
            model=self.models["trend_analysis"],
            symbol=symbol,
            llm_type="technical_synthesis",
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

        # Wrap with metadata for caching
        return self._wrap_llm_response(
            response=response,
            model=self.models["trend_analysis"],
            prompt=prompt,
            temperature=0.3,
            top_p=0.9,
            format="json",
        )

    def _validate_pattern(self, pattern: Dict, price_data: pd.DataFrame) -> bool:
        """Validate detected pattern with price data"""
        # Implement pattern validation logic
        return pattern.get("confidence", 0) > 60

    def _calculate_price_change(self, price_data: pd.DataFrame) -> Dict:
        """Calculate price changes over multiple periods

        Periods aligned with market context analysis:
        - 1d: Daily change
        - 10d: Leading (2 weeks, early warning indicator)
        - 21d: Short-term (1 month, primary sentiment signal)
        - 63d: Medium-term (3 months, quarterly trend)
        - 252d: Long-term (1 year, annual performance)
        """
        # Handle both uppercase and lowercase columns
        close_col = "close" if "close" in price_data.columns else "Close"
        current = price_data[close_col].iloc[-1]

        changes = {}
        # ALIGNED with market context periods: 10d, 21d, 63d, 252d
        periods = [1, 10, 21, 63, 252]  # 1d, 10d (leading), 21d (short), 63d (medium), 252d (long)

        for period in periods:
            if len(price_data) > period:
                prev = price_data[close_col].iloc[-period - 1]
                changes[f"{period}d"] = ((current / prev) - 1) * 100

        return changes

    def _calculate_volume_profile(self, price_data: pd.DataFrame) -> Dict:
        """Calculate volume profile (volume at price levels)"""
        # Handle both uppercase and lowercase columns
        close_col = "close" if "close" in price_data.columns else "Close"
        volume_col = "volume" if "volume" in price_data.columns else "Volume"

        price_bins = pd.qcut(price_data[close_col], q=10)
        volume_profile = price_data.groupby(price_bins, observed=False)[volume_col].sum()

        return {
            "high_volume_node": float(volume_profile.idxmax().mid),
            "low_volume_node": float(volume_profile.idxmin().mid),
            "point_of_control": float(volume_profile.idxmax().mid),
        }

    def _calculate_accumulation_distribution(self, price_data: pd.DataFrame) -> float:
        """Calculate Accumulation/Distribution indicator"""
        # Handle both uppercase and lowercase columns
        close_col = "close" if "close" in price_data.columns else "Close"
        high_col = "high" if "high" in price_data.columns else "High"
        low_col = "low" if "low" in price_data.columns else "Low"
        volume_col = "volume" if "volume" in price_data.columns else "Volume"

        mfm = ((price_data[close_col] - price_data[low_col]) - (price_data[high_col] - price_data[close_col])) / (
            price_data[high_col] - price_data[low_col]
        )
        mfm = mfm.fillna(0)

        ad = (mfm * price_data[volume_col]).cumsum()
        return float(ad.iloc[-1])

    def _find_historical_levels(self, price_data: pd.DataFrame) -> Dict:
        """Find historical support and resistance levels"""
        # Handle both uppercase and lowercase columns
        high_col = "high" if "high" in price_data.columns else "High"
        low_col = "low" if "low" in price_data.columns else "Low"

        # Find local maxima and minima
        window = 20
        local_max = price_data[high_col].rolling(window=window, center=True).max()
        local_min = price_data[low_col].rolling(window=window, center=True).min()

        resistance_levels = price_data[price_data[high_col] == local_max][high_col].unique()
        support_levels = price_data[price_data[low_col] == local_min][low_col].unique()

        resistance_levels = sorted(resistance_levels)
        support_levels = sorted(support_levels)

        return {
            "resistance": [round(float(x), 2) for x in resistance_levels[-3:]] if resistance_levels else [],
            "support": [round(float(x), 2) for x in support_levels[:3]] if support_levels else [],
        }

    def _calculate_fibonacci_levels(self, price_data: pd.DataFrame) -> Dict:
        """Calculate Fibonacci retracement levels"""
        # Handle both uppercase and lowercase columns
        high_col = "high" if "high" in price_data.columns else "High"
        low_col = "low" if "low" in price_data.columns else "Low"

        high = price_data[high_col].max()
        low = price_data[low_col].min()
        diff = high - low

        levels = {
            "0.0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50.0%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "100.0%": low,
        }

        return {k: round(float(v), 2) for k, v in levels.items()}

    def _calculate_momentum_score(self, indicators: Dict) -> float:
        """Calculate overall momentum score from -100 to 100"""
        score = 0
        weight_sum = 0

        # RSI contribution
        if "rsi" in indicators:
            rsi = indicators["rsi"]  # Already a scalar value
            rsi_score = (rsi - 50) * 2  # Convert to -100 to 100
            score += rsi_score * 0.3
            weight_sum += 0.3

        # MACD contribution
        if "macd_histogram" in indicators:
            macd_hist = indicators["macd_histogram"]  # Already a scalar value
            macd_score = np.clip(macd_hist * 100, -100, 100)
            score += macd_score * 0.3
            weight_sum += 0.3

        # Moving average contribution
        if "sma_50" in indicators and "sma_200" in indicators:
            ma_ratio = indicators["sma_50"] / indicators["sma_200"]  # Already scalar values
            ma_score = (ma_ratio - 1) * 100
            score += np.clip(ma_score, -100, 100) * 0.4
            weight_sum += 0.4

        return score / weight_sum if weight_sum > 0 else 0

    def _identify_market_phase(self, price_data: pd.DataFrame) -> str:
        """Identify current market phase"""
        # Handle both uppercase and lowercase columns
        close_col = "close" if "close" in price_data.columns else "Close"

        sma_20 = price_data[close_col].rolling(window=20).mean()
        sma_50 = price_data[close_col].rolling(window=50).mean()

        current_price = price_data[close_col].iloc[-1]

        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            return "uptrend"
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            return "downtrend"
        else:
            return "consolidation"

    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate trend strength (0-100)"""
        # Handle both uppercase and lowercase columns
        close_col = "close" if "close" in price_data.columns else "Close"

        # ADX-like calculation
        returns = price_data[close_col].pct_change(fill_method=None)
        trend_strength = abs(returns.rolling(window=14).mean()) / returns.rolling(window=14).std()

        return float(np.clip(trend_strength.iloc[-1] * 25, 0, 100))

    def _calculate_sentiment_score(self, price_data: pd.DataFrame, volume_analysis: Dict) -> float:
        """Calculate overall sentiment score (-100 to 100)"""
        score = 0

        # Handle both uppercase and lowercase columns
        close_col = "close" if "close" in price_data.columns else "Close"

        # Price trend contribution
        price_change = ((price_data[close_col].iloc[-1] / price_data[close_col].iloc[-20]) - 1) * 100
        score += np.clip(price_change * 2, -50, 50)

        # Volume sentiment contribution
        if volume_analysis.get("accumulation_distribution", 0) > 0:
            score += 25
        else:
            score -= 25

        # Volatility contribution (lower is better)
        returns = price_data[close_col].pct_change(fill_method=None)
        volatility = returns.std()
        if volatility < 0.02:  # Low volatility
            score += 25
        elif volatility > 0.05:  # High volatility
            score -= 25

        return float(np.clip(score, -100, 100))
