#!/usr/bin/env python3
"""
Market Regime Cache Component
Caches market-wide and sector-wide ETF analysis with daily TTL
Reusable across all stock analyses for the same trading day
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from investigator.infrastructure.cache.cache_manager import CacheManager, CacheType
from investigator.infrastructure.database.market_data import get_market_data_fetcher

logger = logging.getLogger(__name__)


class MarketRegimeCache:
    """
    Manages cached market regime analysis that's reusable across stocks

    Components:
    - Market-wide ETF performance (SPY, QQQ, IWM, bonds, commodities)
    - Sector ETF performance (XLK, XLV, XLF, etc.)
    - Market regime classification (risk-on/risk-off/mixed)
    - Volatility indicators (VIX if available)

    All cached with 1-day TTL since we use daily pricing
    """

    def __init__(self, cache_manager: CacheManager = None):
        """Initialize with cache manager"""
        self.cache_manager = cache_manager or CacheManager()
        self.cache_ttl = 86400  # 24 hours in seconds

        # Initialize market data fetcher (uses singleton)
        from investigator.config import get_config

        self.config = get_config()
        self.market_data_fetcher = get_market_data_fetcher(self.config)

        logger.info("Initialized MarketRegimeCache with 24-hour TTL")

    def get_cache_key(self, component_type: str, date: Optional[str] = None) -> Dict:
        """
        Generate cache key for market regime component

        Args:
            component_type: Type of component (market_regime, sector_performance, etc.)
            date: Optional date string (defaults to today)

        Returns:
            Cache key dictionary
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        return {"component_type": component_type, "date": date, "cache_version": "v1.0"}

    def get_market_regime(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get cached market regime analysis or compute if needed

        Args:
            force_refresh: Force recomputation even if cached

        Returns:
            Market regime analysis dict or None
        """
        cache_key = self.get_cache_key("market_regime")

        # Check cache first
        if not force_refresh:
            cached_data = self.cache_manager.get(CacheType.MARKET_REGIME, cache_key)

            if cached_data:
                logger.info(f"✅ Market regime cache HIT for {cache_key['date']}")
                return cached_data

        logger.info(f"🔄 Computing market regime for {cache_key['date']}")

        # Compute market regime
        market_regime = self._compute_market_regime()

        if market_regime:
            # Cache the result
            self.cache_manager.set(CacheType.MARKET_REGIME, cache_key, market_regime, ttl=self.cache_ttl)
            logger.info(f"💾 Cached market regime for {cache_key['date']}")

        return market_regime

    def get_sector_performance(self, sector: Optional[str] = None, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get cached sector performance or compute if needed

        Args:
            sector: Specific sector or None for all sectors
            force_refresh: Force recomputation

        Returns:
            Sector performance dict or None
        """
        cache_key = self.get_cache_key("sector_performance")
        if sector:
            cache_key["sector"] = sector

        # Check cache first
        if not force_refresh:
            cached_data = self.cache_manager.get(CacheType.SECTOR_PERFORMANCE, cache_key)

            if cached_data:
                logger.info(f"✅ Sector performance cache HIT for {cache_key['date']}")
                return cached_data

        logger.info(f"🔄 Computing sector performance for {cache_key['date']}")

        # Compute sector performance
        sector_performance = self._compute_sector_performance(sector)

        if sector_performance:
            # Cache the result
            self.cache_manager.set(CacheType.SECTOR_PERFORMANCE, cache_key, sector_performance, ttl=self.cache_ttl)
            logger.info(f"💾 Cached sector performance for {cache_key['date']}")

        return sector_performance

    def get_market_breadth(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get cached market breadth indicators

        Returns:
            Market breadth analysis (advancing/declining, new highs/lows, etc.)
        """
        cache_key = self.get_cache_key("market_breadth")

        if not force_refresh:
            cached_data = self.cache_manager.get(CacheType.MARKET_BREADTH, cache_key)

            if cached_data:
                logger.info(f"✅ Market breadth cache HIT for {cache_key['date']}")
                return cached_data

        logger.info(f"🔄 Computing market breadth for {cache_key['date']}")

        # Compute market breadth
        market_breadth = self._compute_market_breadth()

        if market_breadth:
            self.cache_manager.set(CacheType.MARKET_BREADTH, cache_key, market_breadth, ttl=self.cache_ttl)
            logger.info(f"💾 Cached market breadth for {cache_key['date']}")

        return market_breadth

    def get_commodity_signals(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get cached commodity and inflation signals

        Returns:
            Commodity performance and inflation indicators
        """
        cache_key = self.get_cache_key("commodity_signals")

        if not force_refresh:
            cached_data = self.cache_manager.get(CacheType.COMMODITY_SIGNALS, cache_key)

            if cached_data:
                logger.info(f"✅ Commodity signals cache HIT for {cache_key['date']}")
                return cached_data

        logger.info(f"🔄 Computing commodity signals for {cache_key['date']}")

        # Compute commodity signals
        commodity_signals = self._compute_commodity_signals()

        if commodity_signals:
            self.cache_manager.set(CacheType.COMMODITY_SIGNALS, cache_key, commodity_signals, ttl=self.cache_ttl)
            logger.info(f"💾 Cached commodity signals for {cache_key['date']}")

        return commodity_signals

    def _compute_market_regime(self) -> Dict:
        """
        Compute comprehensive market regime analysis

        Returns:
            Dict with market regime classification and supporting data
        """
        try:
            regime_data = {
                "computed_at": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "timeframes": {},
            }

            # Define ETFs to analyze
            market_etfs = {
                "broad_market": "SPY",
                "nasdaq": "QQQ",
                "small_cap": "IWM",
                "bonds": "AGG",
                "treasury": "TLT",
                "high_yield": "HYG",
                "gold": "GLD",
                "silver": "SLV",
                "oil": "USO",
                "commodities": "DBC",
                "intl": "EFA",
                "emerging": "EEM",
            }

            # Analyze multiple timeframes
            timeframes = {"short_term": 5, "medium_term": 20, "long_term": 60}  # 1 week  # 1 month  # 3 months

            for tf_name, days in timeframes.items():
                tf_data = {}

                for etf_type, symbol in market_etfs.items():
                    try:
                        # Get ETF data
                        data = self.market_data_fetcher.get_stock_data(symbol, days + 10)

                        if len(data) >= days:
                            # Calculate metrics
                            returns = self._calculate_returns(data, days)
                            volatility = self._calculate_volatility(data, days)
                            momentum = self._calculate_momentum(data)

                            tf_data[etf_type] = {
                                "symbol": symbol,
                                "return": returns,
                                "volatility": volatility,
                                "momentum": momentum,
                                "current_price": float(data["Close"].iloc[-1]),
                            }
                    except Exception as e:
                        logger.warning(f"Failed to analyze {symbol}: {e}")

                regime_data["timeframes"][tf_name] = tf_data

            # Determine overall market regime
            regime_data["regime"] = self._classify_market_regime(regime_data["timeframes"])
            regime_data["risk_signals"] = self._calculate_risk_signals(regime_data["timeframes"])

            return regime_data

        except Exception as e:
            logger.error(f"Failed to compute market regime: {e}")
            return None

    def _compute_sector_performance(self, sector: Optional[str] = None) -> Dict:
        """
        Compute sector performance analysis

        Args:
            sector: Specific sector or None for all

        Returns:
            Sector performance dict
        """
        try:
            sector_etfs = {
                "technology": "XLK",
                "healthcare": "XLV",
                "financials": "XLF",
                "energy": "XLE",
                "industrials": "XLI",
                "consumer_discretionary": "XLY",
                "consumer_staples": "XLP",
                "utilities": "XLU",
                "materials": "XLB",
                "real_estate": "XLRE",
                "communication": "XLC",
            }

            if sector:
                # Filter to specific sector
                sector_etfs = {k: v for k, v in sector_etfs.items() if k == sector}

            performance_data = {
                "computed_at": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "sectors": {},
            }

            for sector_name, etf in sector_etfs.items():
                try:
                    # Get 60 days of data
                    data = self.market_data_fetcher.get_stock_data(etf, 70)

                    if len(data) >= 20:
                        performance_data["sectors"][sector_name] = {
                            "etf": etf,
                            "1w_return": self._calculate_returns(data, 5),
                            "1m_return": self._calculate_returns(data, 20),
                            "3m_return": self._calculate_returns(data, 60) if len(data) >= 60 else None,
                            "volatility": self._calculate_volatility(data, 20),
                            "relative_strength": self._calculate_relative_strength(data, "SPY"),
                            "momentum": self._calculate_momentum(data),
                        }
                except Exception as e:
                    logger.warning(f"Failed to analyze sector {sector_name}: {e}")

            # Rank sectors by performance
            performance_data["rankings"] = self._rank_sectors(performance_data["sectors"])

            return performance_data

        except Exception as e:
            logger.error(f"Failed to compute sector performance: {e}")
            return None

    def _compute_market_breadth(self) -> Dict:
        """
        Compute market breadth indicators

        Returns:
            Market breadth analysis
        """
        try:
            breadth_data = {"computed_at": datetime.now().isoformat(), "date": datetime.now().strftime("%Y-%m-%d")}

            # Analyze broad market vs sector performance
            spy_data = self.market_data_fetcher.get_stock_data("SPY", 30)

            if len(spy_data) >= 20:
                # Count sectors outperforming SPY
                outperforming = 0
                underperforming = 0

                spy_return = self._calculate_returns(spy_data, 20)

                for sector_etf in ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]:
                    try:
                        sector_data = self.market_data_fetcher.get_stock_data(sector_etf, 30)
                        if len(sector_data) >= 20:
                            sector_return = self._calculate_returns(sector_data, 20)
                            if sector_return > spy_return:
                                outperforming += 1
                            else:
                                underperforming += 1
                    except:
                        pass

                breadth_data["sectors_outperforming"] = outperforming
                breadth_data["sectors_underperforming"] = underperforming
                breadth_data["breadth_ratio"] = outperforming / max(outperforming + underperforming, 1)

                # Classify breadth
                if breadth_data["breadth_ratio"] > 0.6:
                    breadth_data["breadth_signal"] = "broad_participation"
                elif breadth_data["breadth_ratio"] < 0.4:
                    breadth_data["breadth_signal"] = "narrow_leadership"
                else:
                    breadth_data["breadth_signal"] = "mixed_breadth"

            return breadth_data

        except Exception as e:
            logger.error(f"Failed to compute market breadth: {e}")
            return None

    def _compute_commodity_signals(self) -> Dict:
        """
        Compute commodity and inflation signals

        Returns:
            Commodity performance indicators
        """
        try:
            commodity_data = {
                "computed_at": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "commodities": {},
            }

            # Analyze key commodities
            commodities = {
                "gold": "GLD",
                "silver": "SLV",
                "oil": "USO",
                "broad_commodities": "DBC",
                "energy_heavy": "GSG",
            }

            for name, symbol in commodities.items():
                try:
                    data = self.market_data_fetcher.get_stock_data(symbol, 70)

                    if len(data) >= 20:
                        commodity_data["commodities"][name] = {
                            "symbol": symbol,
                            "1w_return": self._calculate_returns(data, 5),
                            "1m_return": self._calculate_returns(data, 20),
                            "3m_return": self._calculate_returns(data, 60) if len(data) >= 60 else None,
                            "volatility": self._calculate_volatility(data, 20),
                            "current_price": float(data["Close"].iloc[-1]),
                        }
                except Exception as e:
                    logger.warning(f"Failed to analyze {name}: {e}")

            # Calculate inflation signals
            if "gold" in commodity_data["commodities"] and "broad_commodities" in commodity_data["commodities"]:
                gold_return = commodity_data["commodities"]["gold"]["1m_return"]
                commodity_return = commodity_data["commodities"]["broad_commodities"]["1m_return"]

                # Inflation signal based on commodity performance
                if commodity_return > 0.03 and gold_return > 0.02:
                    commodity_data["inflation_signal"] = "inflationary"
                elif commodity_return < -0.03 and gold_return < -0.02:
                    commodity_data["inflation_signal"] = "deflationary"
                else:
                    commodity_data["inflation_signal"] = "neutral"

            # Calculate energy signal
            if "oil" in commodity_data["commodities"]:
                oil_return = commodity_data["commodities"]["oil"]["1m_return"]
                if oil_return > 0.05:
                    commodity_data["energy_signal"] = "strong_demand"
                elif oil_return < -0.05:
                    commodity_data["energy_signal"] = "weak_demand"
                else:
                    commodity_data["energy_signal"] = "stable"

            return commodity_data

        except Exception as e:
            logger.error(f"Failed to compute commodity signals: {e}")
            return None

    def _calculate_returns(self, data, days):
        """Calculate returns over specified days"""
        if len(data) < days:
            return 0
        start_price = float(data["Close"].iloc[-days])
        end_price = float(data["Close"].iloc[-1])
        return (end_price - start_price) / start_price

    def _calculate_volatility(self, data, days):
        """Calculate annualized volatility"""
        if len(data) < days:
            return 0
        import numpy as np

        recent_data = data.tail(days)
        daily_returns = recent_data["Close"].pct_change().dropna()
        return float(daily_returns.std() * np.sqrt(252))

    def _calculate_momentum(self, data):
        """Calculate momentum indicators"""
        if len(data) < 20:
            return {}

        close_prices = data["Close"]
        current_price = close_prices.iloc[-1]

        # Simple momentum indicators
        sma_20 = close_prices.rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_price
        sma_50 = close_prices.rolling(window=50).mean().iloc[-1] if len(data) >= 50 else current_price

        return {
            "price_vs_sma20": float((current_price - sma_20) / sma_20) if sma_20 > 0 else 0,
            "price_vs_sma50": float((current_price - sma_50) / sma_50) if sma_50 > 0 else 0,
        }

    def _calculate_relative_strength(self, data, benchmark_symbol):
        """Calculate relative strength vs benchmark"""
        try:
            benchmark_data = self.market_data_fetcher.get_stock_data(benchmark_symbol, len(data))
            if len(benchmark_data) >= 20 and len(data) >= 20:
                asset_return = self._calculate_returns(data, 20)
                benchmark_return = self._calculate_returns(benchmark_data, 20)
                return asset_return - benchmark_return
        except:
            return 0

    def _classify_market_regime(self, timeframes):
        """Classify overall market regime based on multi-timeframe analysis"""
        medium_term = timeframes.get("medium_term", {})

        if not medium_term:
            return "neutral"

        risk_signals = 0
        total_signals = 0

        # Check equity vs bond performance
        if "broad_market" in medium_term and "bonds" in medium_term:
            if medium_term["broad_market"]["return"] > medium_term["bonds"]["return"]:
                risk_signals += 1
            total_signals += 1

        # Check small cap vs large cap
        if "small_cap" in medium_term and "broad_market" in medium_term:
            if medium_term["small_cap"]["return"] > medium_term["broad_market"]["return"]:
                risk_signals += 1
            total_signals += 1

        # Check gold performance
        if "gold" in medium_term:
            if medium_term["gold"]["return"] < -0.02:
                risk_signals += 1
            elif medium_term["gold"]["return"] > 0.05:
                risk_signals -= 1
            total_signals += 1

        # Check commodity performance
        if "commodities" in medium_term:
            if medium_term["commodities"]["return"] > 0.03:
                risk_signals += 1
            elif medium_term["commodities"]["return"] < -0.05:
                risk_signals -= 1
            total_signals += 1

        # Calculate risk ratio
        risk_ratio = risk_signals / max(total_signals, 1)

        if risk_ratio >= 0.6:
            return "risk_on"
        elif risk_ratio <= 0.3:
            return "risk_off"
        else:
            return "mixed"

    def _calculate_risk_signals(self, timeframes):
        """Calculate detailed risk signals"""
        signals = {"equity_bond": "neutral", "size_factor": "neutral", "safe_haven": "neutral", "commodity": "neutral"}

        medium_term = timeframes.get("medium_term", {})

        # Equity vs Bond signal
        if "broad_market" in medium_term and "bonds" in medium_term:
            spread = medium_term["broad_market"]["return"] - medium_term["bonds"]["return"]
            if spread > 0.02:
                signals["equity_bond"] = "risk_on"
            elif spread < -0.02:
                signals["equity_bond"] = "risk_off"

        # Size factor signal
        if "small_cap" in medium_term and "broad_market" in medium_term:
            spread = medium_term["small_cap"]["return"] - medium_term["broad_market"]["return"]
            if spread > 0.01:
                signals["size_factor"] = "risk_on"
            elif spread < -0.01:
                signals["size_factor"] = "risk_off"

        # Safe haven signal
        if "gold" in medium_term:
            gold_return = medium_term["gold"]["return"]
            if gold_return > 0.05:
                signals["safe_haven"] = "risk_off"
            elif gold_return < -0.02:
                signals["safe_haven"] = "risk_on"

        # Commodity signal
        if "commodities" in medium_term:
            commodity_return = medium_term["commodities"]["return"]
            if commodity_return > 0.03:
                signals["commodity"] = "inflationary"
            elif commodity_return < -0.05:
                signals["commodity"] = "deflationary"

        return signals

    def _rank_sectors(self, sectors):
        """Rank sectors by performance"""
        rankings = []

        for sector_name, data in sectors.items():
            if "1m_return" in data:
                rankings.append(
                    {
                        "sector": sector_name,
                        "return": data["1m_return"],
                        "relative_strength": data.get("relative_strength", 0),
                    }
                )

        # Sort by return
        rankings.sort(key=lambda x: x["return"], reverse=True)

        # Add rank
        for i, item in enumerate(rankings):
            item["rank"] = i + 1

        return rankings

    def clear_cache(self, date: Optional[str] = None):
        """
        Clear all cached market regime data for a specific date

        Args:
            date: Date to clear (defaults to today)
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        components = ["market_regime", "sector_performance", "market_breadth", "commodity_signals"]

        for component in components:
            cache_key = self.get_cache_key(component, date)
            try:
                # Try to delete from each cache type
                for cache_type in [
                    CacheType.MARKET_REGIME,
                    CacheType.SECTOR_PERFORMANCE,
                    CacheType.MARKET_BREADTH,
                    CacheType.COMMODITY_SIGNALS,
                ]:
                    self.cache_manager.delete(cache_type, cache_key)
            except:
                pass

        logger.info(f"Cleared all market regime cache for {date}")

    def get_cache_stats(self) -> Dict:
        """
        Get statistics about cached market regime data

        Returns:
            Cache statistics
        """
        today = datetime.now().strftime("%Y-%m-%d")
        stats = {"date": today, "cached_components": []}

        # Check what's cached
        components = {
            "market_regime": CacheType.MARKET_REGIME,
            "sector_performance": CacheType.SECTOR_PERFORMANCE,
            "market_breadth": CacheType.MARKET_BREADTH,
            "commodity_signals": CacheType.COMMODITY_SIGNALS,
        }

        for component_name, cache_type in components.items():
            cache_key = self.get_cache_key(component_name)
            if self.cache_manager.get(cache_type, cache_key):
                stats["cached_components"].append(component_name)

        stats["cache_coverage"] = len(stats["cached_components"]) / len(components) * 100

        return stats
