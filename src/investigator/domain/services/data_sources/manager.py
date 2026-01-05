"""
DataSourceManager - Unified Facade for All Data Sources

Provides a single entry point for accessing all data sources with:
- Automatic source registration and discovery
- Parallel data fetching
- Caching and fallback strategies
- Consistent data format for consumers (CLI, backtest, batch runner)
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set

from .base import DataCategory, DataQuality, DataResult, DataSource
from .registry import DataSourceRegistry, get_registry

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedData:
    """
    Consolidated data from all sources for a symbol.

    This is the primary output format used by:
    - CLI analysis commands
    - RL backtest
    - Batch analysis runner
    - Valuation models
    """

    symbol: str
    as_of_date: date
    timestamp: datetime = field(default_factory=datetime.now)

    # Market Data
    price: Optional[Dict[str, Any]] = None
    technical: Optional[Dict[str, Any]] = None

    # Fundamental Data
    financials: Optional[Dict[str, Any]] = None
    ratios: Optional[Dict[str, Any]] = None

    # Sentiment Data
    insider: Optional[Dict[str, Any]] = None
    institutional: Optional[Dict[str, Any]] = None
    short_interest: Optional[Dict[str, Any]] = None

    # Macro Data
    macro: Optional[Dict[str, Any]] = None
    fed_districts: Optional[Dict[str, Any]] = None
    treasury: Optional[Dict[str, Any]] = None

    # Volatility
    volatility: Optional[Dict[str, Any]] = None

    # Quality Metrics
    sources_succeeded: List[str] = field(default_factory=list)
    sources_failed: List[str] = field(default_factory=list)
    overall_quality: DataQuality = DataQuality.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "as_of_date": self.as_of_date.isoformat(),
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "technical": self.technical,
            "financials": self.financials,
            "ratios": self.ratios,
            "insider": self.insider,
            "institutional": self.institutional,
            "short_interest": self.short_interest,
            "macro": self.macro,
            "fed_districts": self.fed_districts,
            "treasury": self.treasury,
            "volatility": self.volatility,
            "quality": {
                "succeeded": self.sources_succeeded,
                "failed": self.sources_failed,
                "overall": self.overall_quality.name,
            },
        }

    def get_rl_features(self) -> Dict[str, float]:
        """
        Extract features for RL model.

        Returns a flat dictionary of numeric features suitable for RL training.
        Features are normalized or scaled to useful ranges for ML models:
        - Percentages/ratios: typically 0-100 or 0-1
        - Rates/yields: raw values (small positive numbers)
        - Binary flags: 0.0 or 1.0
        - Scores: 0-100 range
        """
        features = {}

        # =================================================================
        # 1. Price/Return Features
        # =================================================================
        if self.price and self.price.get("returns"):
            # Returns are already in percentage form
            features["return_1d"] = self.price["returns"].get("1d", 0.0)
            features["return_5d"] = self.price["returns"].get("5d", 0.0)
            features["return_1m"] = self.price["returns"].get("1m", 0.0)

        # =================================================================
        # 2. Technical Features
        # =================================================================
        if self.technical:
            indicators = self.technical.get("indicators", {})
            # RSI is already 0-100
            features["rsi"] = float(indicators.get("rsi_14", 50))
            # Binary flags
            features["above_sma_20"] = 1.0 if indicators.get("above_sma_20") else 0.0
            features["above_sma_50"] = 1.0 if indicators.get("above_sma_50") else 0.0

        # =================================================================
        # 3. Treasury/Yield Curve Features
        # =================================================================
        if self.treasury:
            yields = self.treasury.get("yields", {})

            # 10Y yield level (key benchmark rate)
            yield_10y = yields.get("10Y", {}).get("rate")
            if yield_10y is not None:
                features["yield_10y"] = float(yield_10y)

            # Yield curve slope (10Y-2Y spread) - key recession indicator
            # Positive = normal curve, negative = inverted
            spread_10y_2y = self.treasury.get("spread_10y_2y")
            if spread_10y_2y is not None:
                features["yield_curve_slope"] = float(spread_10y_2y)

            # Inversion flag (binary)
            is_inverted = self.treasury.get("is_inverted_10_2")
            if is_inverted is not None:
                features["yield_curve_inverted"] = 1.0 if is_inverted else 0.0

        # =================================================================
        # 4. Volatility/CBOE Features
        # =================================================================
        if self.volatility:
            # VIX level (typically 10-80, mean ~18)
            vix = self.volatility.get("vix")
            if vix is not None:
                features["vix"] = float(vix)
                # Normalize VIX to 0-100 percentile-like scale
                # VIX < 12 = 0-10%, VIX 12-20 = 10-50%, VIX 20-30 = 50-80%, VIX > 30 = 80-100%
                features["vix_normalized"] = self._normalize_vix(float(vix))

            # SKEW index (tail risk indicator, typically 100-160, mean ~125)
            skew = self.volatility.get("skew")
            if skew is not None:
                features["skew"] = float(skew)
                # Normalize SKEW: 100=0, 130=50, 160=100
                features["skew_normalized"] = max(0.0, min(100.0, (float(skew) - 100) / 0.6))

            # Fear/greed score (already 0-100)
            fear_greed = self.volatility.get("fear_greed_score")
            if fear_greed is not None:
                features["fear_greed_score"] = float(fear_greed)

            # VIX term structure (contango vs backwardation)
            term_ratio = self.volatility.get("term_structure_ratio")
            if term_ratio is not None:
                features["vix_term_structure"] = float(term_ratio)
            is_backwardation = self.volatility.get("is_backwardation")
            if is_backwardation is not None:
                features["vix_backwardation"] = 1.0 if is_backwardation else 0.0

        # =================================================================
        # 5. Fed District Features (GDPNow, CFNAI, etc.)
        # =================================================================
        if self.fed_districts:
            summary = self.fed_districts.get("summary", {})

            # GDPNow (Atlanta Fed real-time GDP estimate, typically -5 to +10)
            gdpnow = summary.get("gdpnow")
            if gdpnow is not None:
                features["gdpnow"] = float(gdpnow)

            # CFNAI (Chicago Fed National Activity Index)
            # Zero = trend growth, positive = above trend, negative = below
            # Typically ranges from -4 to +1
            cfnai = summary.get("cfnai")
            if cfnai is not None:
                features["cfnai"] = float(cfnai)

            # NFCI (Chicago Fed Financial Conditions Index)
            # Zero = average, positive = tighter than average
            nfci = summary.get("nfci")
            if nfci is not None:
                features["nfci"] = float(nfci)

            # NY Fed Recession Probability (0-100%)
            recession_prob = summary.get("recession_probability")
            if recession_prob is not None:
                features["recession_probability"] = float(recession_prob)

            # Look in by_district if summary doesn't have the values
            by_district = self.fed_districts.get("by_district", {})

            # Atlanta Fed
            atlanta = by_district.get("atlanta", {})
            if "gdpnow" not in features:
                gdpnow_data = atlanta.get("gdpnow", {})
                if gdpnow_data and gdpnow_data.get("value") is not None:
                    features["gdpnow"] = float(gdpnow_data["value"])

            # Chicago Fed
            chicago = by_district.get("chicago", {})
            if "cfnai" not in features:
                cfnai_data = chicago.get("cfnai", {})
                if cfnai_data and cfnai_data.get("value") is not None:
                    features["cfnai"] = float(cfnai_data["value"])

        # =================================================================
        # 6. Short Interest Features
        # =================================================================
        if self.short_interest:
            current = self.short_interest.get("current", {})

            # Days to cover (typically 1-20, higher = more squeeze potential)
            days_to_cover = current.get("days_to_cover")
            if days_to_cover is not None:
                features["days_to_cover"] = float(days_to_cover)
                # Normalize: cap at 20 days, scale to 0-100
                features["days_to_cover_normalized"] = min(100.0, float(days_to_cover) * 5)

            # Short percent of float (typically 1-50%)
            short_pct = current.get("short_pct_float")
            if short_pct is not None:
                features["short_pct_float"] = float(short_pct)

        # =================================================================
        # 7. Macro Economic Features
        # =================================================================
        if self.macro:
            # GDP data
            gdp = self.macro.get("gdp", {})

            # Unemployment rate (typically 3-10%)
            unrate_data = gdp.get("UNRATE", {})
            if unrate_data and unrate_data.get("value") is not None:
                features["unemployment_rate"] = float(unrate_data["value"])

            # Inflation data
            inflation = self.macro.get("inflation", {})

            # 5-Year breakeven inflation (market inflation expectations)
            t5yie_data = inflation.get("T5YIE", {})
            if t5yie_data and t5yie_data.get("value") is not None:
                features["inflation_expectations_5y"] = float(t5yie_data["value"])

            # 10-Year breakeven inflation
            t10yie_data = inflation.get("T10YIE", {})
            if t10yie_data and t10yie_data.get("value") is not None:
                features["inflation_expectations_10y"] = float(t10yie_data["value"])

            # Core PCE (Fed's preferred inflation measure)
            pcepilfe_data = inflation.get("PCEPILFE", {})
            if pcepilfe_data and pcepilfe_data.get("value") is not None:
                features["core_pce"] = float(pcepilfe_data["value"])

            # Fed Funds Rate
            rates = self.macro.get("rates", {})
            dff_data = rates.get("DFF", {})
            if dff_data and dff_data.get("value") is not None:
                features["fed_funds_rate"] = float(dff_data["value"])

            # Credit spreads
            spreads = self.macro.get("spreads", {})

            # High yield spread (risk appetite indicator)
            hy_spread = spreads.get("BAMLH0A0HYM2", {})
            if hy_spread and hy_spread.get("value") is not None:
                features["high_yield_spread"] = float(hy_spread["value"])

            # Consumer sentiment (Michigan)
            consumer = self.macro.get("consumer", {})
            umcsent_data = consumer.get("UMCSENT", {})
            if umcsent_data and umcsent_data.get("value") is not None:
                features["consumer_sentiment"] = float(umcsent_data["value"])

        # =================================================================
        # 8. Insider Trading Features
        # =================================================================
        if self.insider:
            summary = self.insider.get("summary", {})
            features["insider_buys"] = float(summary.get("buys", 0))
            features["insider_sells"] = float(summary.get("sells", 0))
            # Net insider activity (positive = more buying)
            features["insider_net"] = features.get("insider_buys", 0) - features.get("insider_sells", 0)

        return features

    def _normalize_vix(self, vix: float) -> float:
        """
        Normalize VIX to a 0-100 percentile-like scale.

        Based on historical VIX distribution:
        - VIX < 12: Extremely low (complacency) - 0-10
        - VIX 12-15: Low - 10-25
        - VIX 15-18: Below average - 25-40
        - VIX 18-22: Average - 40-60
        - VIX 22-28: Elevated - 60-80
        - VIX 28-35: High - 80-90
        - VIX > 35: Extreme fear - 90-100
        """
        if vix < 12:
            return vix / 12 * 10
        elif vix < 15:
            return 10 + (vix - 12) / 3 * 15
        elif vix < 18:
            return 25 + (vix - 15) / 3 * 15
        elif vix < 22:
            return 40 + (vix - 18) / 4 * 20
        elif vix < 28:
            return 60 + (vix - 22) / 6 * 20
        elif vix < 35:
            return 80 + (vix - 28) / 7 * 10
        else:
            return min(100.0, 90 + (vix - 35) / 15 * 10)


class DataSourceManager:
    """
    Unified facade for all data sources.

    Usage:
        manager = DataSourceManager()

        # Get all data for a symbol
        data = manager.get_data("AAPL")

        # Get specific categories
        macro = manager.get_macro_data()
        sentiment = manager.get_sentiment_data("AAPL")

        # Batch fetch
        batch_data = manager.get_batch_data(["AAPL", "MSFT", "GOOGL"])
    """

    _instance: Optional["DataSourceManager"] = None

    def __new__(cls) -> "DataSourceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._registry = get_registry()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._logger = logging.getLogger("DataSourceManager")
        self._initialize_sources()

    def _initialize_sources(self) -> None:
        """Initialize and register all data sources"""
        # Import sources to trigger registration
        from .sources import (
            AtlantaFedSource,
            CBOEVolatilitySource,
            ChicagoFedSource,
            ClevelandFedSource,
            DallasFedSource,
            FredMacroSource,
            InsiderTransactionSource,
            InstitutionalHoldingsSource,
            KansasCityFedSource,
            NewYorkFedSource,
            PhiladelphiaFedSource,
            PriceHistorySource,
            RichmondFedSource,
            SECQuarterlySource,
            ShortInterestSource,
            TechnicalIndicatorSource,
            TreasuryYieldSource,
        )

        self._logger.info(f"Initialized {len(self._registry.list_sources())} data sources")

    def get_data(
        self, symbol: str, as_of_date: Optional[date] = None, categories: Optional[List[DataCategory]] = None
    ) -> ConsolidatedData:
        """
        Get consolidated data from all sources for a symbol.

        Args:
            symbol: Stock symbol
            as_of_date: Historical date (default: today)
            categories: Specific categories to fetch (default: all)

        Returns:
            ConsolidatedData with all available data
        """
        target_date = as_of_date or date.today()
        data = ConsolidatedData(symbol=symbol, as_of_date=target_date)

        # Define source mappings
        source_mapping = {
            "price_history": ("price", DataCategory.MARKET_DATA),
            "technical_indicators": ("technical", DataCategory.MARKET_DATA),
            "sec_quarterly": ("financials", DataCategory.FUNDAMENTAL),
            "insider_transactions": ("insider", DataCategory.SENTIMENT),
            "institutional_holdings": ("institutional", DataCategory.SENTIMENT),
            "short_interest": ("short_interest", DataCategory.SENTIMENT),
            "fred_macro": ("macro", DataCategory.MACRO),
            "all_fed_districts": ("fed_districts", DataCategory.MACRO),
            "treasury_yields": ("treasury", DataCategory.FIXED_INCOME),
            "cboe_volatility": ("volatility", DataCategory.VOLATILITY),
        }

        # Filter by categories if specified
        if categories:
            source_mapping = {k: v for k, v in source_mapping.items() if v[1] in categories}

        # Fetch data from each source
        for source_name, (attr_name, category) in source_mapping.items():
            source = self._registry.get(source_name)
            if not source:
                continue

            try:
                # Use _MACRO for macro sources
                fetch_symbol = "_MACRO" if category == DataCategory.MACRO else symbol
                result = source.fetch(fetch_symbol, target_date)

                if result.success:
                    setattr(data, attr_name, result.data)
                    data.sources_succeeded.append(source_name)
                else:
                    data.sources_failed.append(source_name)
                    self._logger.debug(f"{source_name} failed: {result.error}")

            except Exception as e:
                data.sources_failed.append(source_name)
                self._logger.error(f"{source_name} error: {e}")

        # Calculate overall quality
        success_rate = len(data.sources_succeeded) / max(len(source_mapping), 1)
        if success_rate >= 0.8:
            data.overall_quality = DataQuality.HIGH
        elif success_rate >= 0.5:
            data.overall_quality = DataQuality.MEDIUM
        else:
            data.overall_quality = DataQuality.LOW

        return data

    def get_macro_data(self, as_of_date: Optional[date] = None) -> Dict[str, Any]:
        """Get all macro economic data"""
        target_date = as_of_date or date.today()
        data = {}

        macro_sources = [
            "fred_macro",
            "all_fed_districts",
            "treasury_yields",
            "cboe_volatility",
        ]

        for source_name in macro_sources:
            source = self._registry.get(source_name)
            if source:
                result = source.fetch("_MACRO", target_date)
                if result.success:
                    data[source_name] = result.data

        return data

    def get_sentiment_data(self, symbol: str, as_of_date: Optional[date] = None) -> Dict[str, Any]:
        """Get sentiment data for a symbol"""
        target_date = as_of_date or date.today()
        data = {}

        sentiment_sources = [
            "insider_transactions",
            "institutional_holdings",
            "short_interest",
        ]

        for source_name in sentiment_sources:
            source = self._registry.get(source_name)
            if source:
                result = source.fetch(symbol, target_date)
                if result.success:
                    data[source_name] = result.data

        return data

    def get_batch_data(
        self, symbols: List[str], as_of_date: Optional[date] = None, parallel: bool = True
    ) -> Dict[str, ConsolidatedData]:
        """
        Get data for multiple symbols.

        Args:
            symbols: List of stock symbols
            as_of_date: Historical date
            parallel: Use parallel fetching

        Returns:
            Dictionary mapping symbol to ConsolidatedData
        """
        target_date = as_of_date or date.today()

        if parallel:
            # Parallel fetch using thread pool
            futures = {symbol: self._executor.submit(self.get_data, symbol, target_date) for symbol in symbols}
            results = {symbol: future.result() for symbol, future in futures.items()}
        else:
            # Sequential fetch
            results = {symbol: self.get_data(symbol, target_date) for symbol in symbols}

        return results

    def get_rl_context(self, symbol: str, as_of_date: Optional[date] = None) -> Dict[str, float]:
        """
        Get features for RL model.

        Returns a flat dictionary of numeric features suitable for RL.
        """
        data = self.get_data(symbol, as_of_date)
        return data.get_rl_features()

    def list_sources(self) -> List[Dict[str, Any]]:
        """List all available data sources"""
        return self._registry.list_sources()

    def get_source(self, name: str) -> Optional[DataSource]:
        """Get a specific data source"""
        return self._registry.get(name)

    def refresh_source(self, name: str, symbol: Optional[str] = None) -> bool:
        """Refresh/invalidate cache for a source"""
        source = self._registry.get(name)
        if source:
            source.invalidate_cache(symbol)
            return True
        return False

    def health_check(self) -> Dict[str, Any]:
        """Check health of all sources"""
        from .registry import check_all_sources_health

        return {
            "sources": check_all_sources_health(),
            "total": len(self._registry.list_sources()),
        }


# =============================================================================
# Global Access Functions
# =============================================================================

_manager: Optional[DataSourceManager] = None


def get_data_source_manager() -> DataSourceManager:
    """Get the global DataSourceManager instance"""
    global _manager
    if _manager is None:
        _manager = DataSourceManager()
    return _manager


def get_consolidated_data(symbol: str, as_of_date: Optional[date] = None) -> ConsolidatedData:
    """Convenience function to get consolidated data"""
    return get_data_source_manager().get_data(symbol, as_of_date)


def get_macro_indicators(as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """Convenience function to get macro data"""
    return get_data_source_manager().get_macro_data(as_of_date)


def get_rl_features(symbol: str, as_of_date: Optional[date] = None) -> Dict[str, float]:
    """Convenience function to get RL features"""
    return get_data_source_manager().get_rl_context(symbol, as_of_date)
