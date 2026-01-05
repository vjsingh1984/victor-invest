"""
Market Data Sources

Provides price history, technical indicators, and short interest data.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from ..base import DataCategory, DataFrequency, DataQuality, DataResult, DataSource, MarketDataSource, SourceMetadata
from ..registry import register_source

logger = logging.getLogger(__name__)


@register_source("price_history", DataCategory.MARKET_DATA)
class PriceHistorySource(MarketDataSource):
    """
    Historical Price Data Source

    Provides:
    - OHLCV data
    - Split-adjusted prices
    - Return calculations
    """

    def __init__(self):
        super().__init__("price_history", DataFrequency.DAILY)

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="price_history",
            category=DataCategory.MARKET_DATA,
            frequency=DataFrequency.DAILY,
            description="Historical OHLCV price data",
            provider="Yahoo Finance / Database",
            is_free=True,
            requires_api_key=False,
            lookback_days=365 * 20,
            symbols_supported=True,
        )

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Fetch current price and recent history using PriceService"""
        try:
            from investigator.domain.services.market_data.price_service import PriceService

            service = PriceService()
            target_date = as_of_date or date.today()
            start_date = target_date - timedelta(days=30)

            # Get price history as DataFrame
            df = service.get_price_history(symbol, start_date, target_date)

            if df is None or df.empty:
                return DataResult(
                    success=False,
                    error=f"No price data for {symbol}",
                    source=self.name,
                )

            # Convert to list of dicts
            prices = []
            for idx, row in df.iterrows():
                prices.append(
                    {
                        "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                        "open": float(row.get("open", 0)) if row.get("open") else None,
                        "high": float(row.get("high", 0)) if row.get("high") else None,
                        "low": float(row.get("low", 0)) if row.get("low") else None,
                        "close": float(row.get("close", 0)) if row.get("close") else None,
                        "volume": int(row.get("volume", 0)) if row.get("volume") else None,
                        "adj_close": float(row.get("adj_close", 0)) if row.get("adj_close") else None,
                    }
                )

            # Reverse so most recent is first
            prices = prices[::-1]

            if not prices:
                return DataResult(
                    success=False,
                    error=f"No price data for {symbol}",
                    source=self.name,
                )

            latest = prices[0]

            # Calculate returns
            returns = {}
            if len(prices) >= 2 and prices[0]["close"] and prices[1]["close"]:
                returns["1d"] = (prices[0]["close"] - prices[1]["close"]) / prices[1]["close"] * 100
            if len(prices) >= 6 and prices[0]["close"] and prices[5]["close"]:
                returns["5d"] = (prices[0]["close"] - prices[5]["close"]) / prices[5]["close"] * 100
            if len(prices) >= 22 and prices[0]["close"] and prices[21]["close"]:
                returns["1m"] = (prices[0]["close"] - prices[21]["close"]) / prices[21]["close"] * 100

            return DataResult(
                success=True,
                data={
                    "symbol": symbol,
                    "current": latest,
                    "prices": prices[:10],  # Last 10 days
                    "returns": returns,
                },
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            logger.error(f"Price history fetch error: {e}")
            return DataResult(success=False, error=str(e), source=self.name)

    def fetch_historical(self, symbol: str, start_date: date, end_date: date) -> DataResult:
        """Fetch historical price range using PriceService"""
        try:
            from investigator.domain.services.market_data.price_service import PriceService

            service = PriceService()
            df = service.get_price_history(symbol, start_date, end_date)

            if df is None or df.empty:
                return DataResult(
                    success=False,
                    error=f"No price data for {symbol} in range",
                    source=self.name,
                )

            prices = []
            for idx, row in df.iterrows():
                prices.append(
                    {
                        "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                        "open": float(row.get("open", 0)) if row.get("open") else None,
                        "high": float(row.get("high", 0)) if row.get("high") else None,
                        "low": float(row.get("low", 0)) if row.get("low") else None,
                        "close": float(row.get("close", 0)) if row.get("close") else None,
                        "volume": int(row.get("volume", 0)) if row.get("volume") else None,
                        "adj_close": float(row.get("adj_close", 0)) if row.get("adj_close") else None,
                    }
                )

            return DataResult(
                success=True,
                data={
                    "symbol": symbol,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "count": len(prices),
                    "prices": prices,
                },
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)


@register_source("technical_indicators", DataCategory.MARKET_DATA)
class TechnicalIndicatorSource(MarketDataSource):
    """
    Technical Indicator Data Source

    Provides:
    - RSI, MACD, Bollinger Bands
    - Moving averages (SMA, EMA)
    - Volume indicators (OBV, MFI)
    - Momentum indicators
    """

    def __init__(self):
        super().__init__("technical_indicators", DataFrequency.DAILY)

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="technical_indicators",
            category=DataCategory.MARKET_DATA,
            frequency=DataFrequency.DAILY,
            description="Technical analysis indicators",
            provider="Computed",
            is_free=True,
            requires_api_key=False,
            lookback_days=365,
            symbols_supported=True,
        )

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Fetch technical indicators"""
        try:
            from investigator.domain.services.market_data.technical_analysis_service import TechnicalAnalysisService

            service = TechnicalAnalysisService()
            target_date = as_of_date or date.today()
            start_date = target_date - timedelta(days=100)

            # Pass both start_date and end_date
            indicators = service.calculate_indicators(symbol, start_date, target_date)

            # Handle DataFrame or empty result
            import pandas as pd

            if isinstance(indicators, pd.DataFrame):
                if indicators.empty:
                    return DataResult(
                        success=False,
                        error=f"Could not calculate indicators for {symbol}",
                        source=self.name,
                    )
                # Get last row of DataFrame as dict
                indicators_dict = indicators.iloc[-1].to_dict()
            elif indicators is None:
                return DataResult(
                    success=False,
                    error=f"Could not calculate indicators for {symbol}",
                    source=self.name,
                )
            elif hasattr(indicators, "__dict__"):
                indicators_dict = {k: v for k, v in indicators.__dict__.items() if not k.startswith("_")}
            elif hasattr(indicators, "to_dict"):
                indicators_dict = indicators.to_dict()
            else:
                indicators_dict = dict(indicators) if indicators else {}

            # Extract key signals
            signals = self._extract_signals(indicators_dict)

            return DataResult(
                success=True,
                data={
                    "symbol": symbol,
                    "indicators": indicators_dict,
                    "signals": signals,
                },
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except ImportError:
            # Fallback to basic calculation
            return self._calculate_basic_indicators(symbol, as_of_date)
        except Exception as e:
            logger.error(f"Technical indicator fetch error: {e}")
            return DataResult(success=False, error=str(e), source=self.name)

    def _calculate_basic_indicators(self, symbol: str, as_of_date: Optional[date]) -> DataResult:
        """Calculate basic indicators from price data using PriceService"""
        try:
            from investigator.domain.services.market_data.price_service import PriceService

            service = PriceService()
            target_date = as_of_date or date.today()
            start_date = target_date - timedelta(days=100)

            df = service.get_price_history(symbol, start_date, target_date)

            if df is None or len(df) < 20:
                return DataResult(
                    success=False,
                    error=f"Insufficient data for {symbol}",
                    source=self.name,
                )

            closes = df["close"].tolist()

            # Calculate basic indicators
            indicators = {}

            # SMA
            indicators["sma_20"] = sum(closes[-20:]) / 20
            indicators["sma_50"] = sum(closes[-50:]) / 50 if len(closes) >= 50 else None

            # RSI (14-day)
            changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
            gains = [c if c > 0 else 0 for c in changes[-14:]]
            losses = [-c if c < 0 else 0 for c in changes[-14:]]
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                indicators["rsi_14"] = 100 - (100 / (1 + rs))
            else:
                indicators["rsi_14"] = 100

            # Current price vs SMAs
            current = closes[-1]
            indicators["current_price"] = current
            indicators["above_sma_20"] = current > indicators["sma_20"]
            if indicators["sma_50"]:
                indicators["above_sma_50"] = current > indicators["sma_50"]

            return DataResult(
                success=True,
                data={
                    "symbol": symbol,
                    "indicators": indicators,
                    "signals": self._extract_signals(indicators),
                },
                source=self.name,
                quality=DataQuality.MEDIUM,
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)

    def _extract_signals(self, indicators: Dict) -> Dict[str, Any]:
        """Extract trading signals from indicators"""
        signals = {}

        rsi = indicators.get("rsi_14") or indicators.get("rsi")
        if rsi:
            if rsi > 70:
                signals["rsi_signal"] = "overbought"
            elif rsi < 30:
                signals["rsi_signal"] = "oversold"
            else:
                signals["rsi_signal"] = "neutral"

        if indicators.get("above_sma_20") and indicators.get("above_sma_50"):
            signals["trend"] = "bullish"
        elif not indicators.get("above_sma_20") and not indicators.get("above_sma_50"):
            signals["trend"] = "bearish"
        else:
            signals["trend"] = "mixed"

        return signals

    def fetch_historical(self, symbol: str, start_date: date, end_date: date) -> DataResult:
        """Historical indicators not supported - use current only"""
        return self._fetch_impl(symbol, end_date)


@register_source("short_interest", DataCategory.SENTIMENT)
class ShortInterestSource(DataSource):
    """
    Short Interest Data Source

    Provides:
    - Short interest shares
    - Days to cover
    - Short squeeze potential
    """

    def __init__(self):
        super().__init__("short_interest", DataCategory.SENTIMENT, DataFrequency.BIWEEKLY)

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="short_interest",
            category=DataCategory.SENTIMENT,
            frequency=DataFrequency.BIWEEKLY,
            description="FINRA short interest data",
            provider="FINRA",
            is_free=True,
            requires_api_key=False,
            lookback_days=365,
            symbols_supported=True,
        )

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Fetch short interest data"""
        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            engine = get_db_manager().engine

            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                        SELECT
                            settlement_date,
                            short_interest,
                            avg_daily_volume,
                            days_to_cover,
                            short_interest_ratio
                        FROM short_interest
                        WHERE symbol = :symbol
                        ORDER BY settlement_date DESC
                        LIMIT 10
                    """
                    ),
                    {"symbol": symbol},
                )

                history = []
                for row in result:
                    history.append(
                        {
                            "date": row[0].isoformat() if row[0] else None,
                            "short_interest": int(row[1]) if row[1] else None,
                            "avg_volume": int(row[2]) if row[2] else None,
                            "days_to_cover": float(row[3]) if row[3] else None,
                            "short_pct_float": float(row[4]) * 100 if row[4] else None,  # Convert ratio to percent
                        }
                    )

            if not history:
                return DataResult(
                    success=False,
                    error=f"No short interest data for {symbol}",
                    source=self.name,
                )

            latest = history[0]

            # Analyze squeeze potential
            squeeze_risk = "low"
            if latest.get("days_to_cover") and latest["days_to_cover"] > 10:
                squeeze_risk = "high"
            elif latest.get("days_to_cover") and latest["days_to_cover"] > 5:
                squeeze_risk = "moderate"

            if latest.get("short_pct_float") and latest["short_pct_float"] > 20:
                squeeze_risk = "high"
            elif latest.get("short_pct_float") and latest["short_pct_float"] > 10:
                if squeeze_risk != "high":
                    squeeze_risk = "moderate"

            # Calculate trend
            trend = "stable"
            if len(history) >= 2:
                if history[0].get("short_interest") and history[1].get("short_interest"):
                    change = (history[0]["short_interest"] - history[1]["short_interest"]) / history[1][
                        "short_interest"
                    ]
                    if change > 0.1:
                        trend = "increasing"
                    elif change < -0.1:
                        trend = "decreasing"

            return DataResult(
                success=True,
                data={
                    "symbol": symbol,
                    "current": latest,
                    "history": history,
                    "analysis": {
                        "squeeze_risk": squeeze_risk,
                        "trend": trend,
                    },
                },
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            logger.error(f"Short interest fetch error: {e}")
            return DataResult(success=False, error=str(e), source=self.name)
