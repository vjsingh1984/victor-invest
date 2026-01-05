# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified Data Source Facade.

Provides a single entry point for accessing all data sources with:
- Consistent interface across CLI, backtest, and batch processing
- Lazy loading of data sources
- Caching and batching for efficiency
- Historical data support for backtesting
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from investigator.config.lookback_periods import (
    CREDIT_RISK_PERIODS,
    INSIDER_PERIODS,
    INSTITUTIONAL_PERIODS,
    MACRO_PERIODS,
    SHORT_INTEREST_PERIODS,
    TECHNICAL_PERIODS,
)
from investigator.domain.services.data_sources.interfaces import (
    DataSourceResult,
    DataSourceType,
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisData:
    """Consolidated data for analysis/backtesting.

    Contains all data sources needed for RL feature extraction.
    """

    symbol: str
    as_of_date: date
    # Core financial data (from existing pipelines)
    financials: Dict[str, Any] = field(default_factory=dict)
    ratios: Dict[str, Any] = field(default_factory=dict)
    # Market context
    market_context: Dict[str, Any] = field(default_factory=dict)
    technical_indicators: Dict[str, Any] = field(default_factory=dict)
    entry_exit_signals: Dict[str, Any] = field(default_factory=dict)
    # Sentiment and ownership
    insider_data: Dict[str, Any] = field(default_factory=dict)
    institutional_data: Dict[str, Any] = field(default_factory=dict)
    short_interest: Dict[str, Any] = field(default_factory=dict)
    # Macro context
    macro_indicators: Dict[str, Any] = field(default_factory=dict)
    market_regime: Dict[str, Any] = field(default_factory=dict)
    treasury_data: Dict[str, Any] = field(default_factory=dict)
    # Regional Fed economic indicators (GDPNow, CFNAI, etc.)
    regional_fed_indicators: Dict[str, Any] = field(default_factory=dict)
    # CBOE volatility data (VIX, SKEW, term structure)
    cboe_data: Dict[str, Any] = field(default_factory=dict)
    # Credit risk
    credit_risk: Dict[str, Any] = field(default_factory=dict)
    # Data quality
    data_quality: Dict[str, Any] = field(default_factory=dict)
    # Price data
    current_price: Optional[float] = None
    price_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "as_of_date": self.as_of_date.isoformat(),
            "financials": self.financials,
            "ratios": self.ratios,
            "market_context": self.market_context,
            "technical_indicators": self.technical_indicators,
            "entry_exit_signals": self.entry_exit_signals,
            "insider_data": self.insider_data,
            "institutional_data": self.institutional_data,
            "short_interest": self.short_interest,
            "macro_indicators": self.macro_indicators,
            "market_regime": self.market_regime,
            "treasury_data": self.treasury_data,
            "regional_fed_indicators": self.regional_fed_indicators,
            "cboe_data": self.cboe_data,
            "credit_risk": self.credit_risk,
            "data_quality": self.data_quality,
            "current_price": self.current_price,
        }


class DataSourceFacade:
    """Unified facade for all data sources.

    Provides consistent access to data for:
    - CLI analysis: get_analysis_data(symbol)
    - Backtesting: get_historical_data(symbol, as_of_date)
    - Batch processing: get_batch_data(symbols)

    Usage:
        facade = get_data_source_facade()

        # For CLI/real-time analysis
        data = await facade.get_analysis_data("AAPL")

        # For backtesting at historical date
        data = await facade.get_historical_data("AAPL", date(2024, 6, 15))

        # For batch processing
        all_data = await facade.get_batch_data(["AAPL", "MSFT", "GOOGL"])
    """

    def __init__(self, db_session=None):
        """Initialize facade with optional database session.

        Args:
            db_session: Optional SQLAlchemy session for database queries.
        """
        self._db_session = db_session
        self._cache: Dict[str, Dict[date, AnalysisData]] = {}
        self._cache_ttl = timedelta(minutes=15)
        self._last_cache_clean = datetime.now()
        self._stock_engine = None  # Lazy-initialized engine for stock database

    async def get_analysis_data(
        self,
        symbol: str,
        include_sources: Optional[Set[DataSourceType]] = None,
    ) -> AnalysisData:
        """Get comprehensive analysis data for a symbol.

        Args:
            symbol: Stock ticker symbol.
            include_sources: Optional set of sources to include.
                           If None, includes all available sources.

        Returns:
            AnalysisData with all requested data sources populated.
        """
        return self.get_historical_data_sync(
            symbol=symbol,
            as_of_date=date.today(),
            include_sources=include_sources,
        )

    def get_historical_data_sync(
        self,
        symbol: str,
        as_of_date: date,
        include_sources: Optional[Set[DataSourceType]] = None,
    ) -> AnalysisData:
        """Get analysis data as of a historical date (synchronous).

        Args:
            symbol: Stock ticker symbol.
            as_of_date: Date to get data as of.
            include_sources: Optional set of sources to include.

        Returns:
            AnalysisData with historical data populated.
        """
        # Check cache
        cache_key = f"{symbol}_{as_of_date.isoformat()}"
        if symbol in self._cache and as_of_date in self._cache[symbol]:
            cached = self._cache[symbol][as_of_date]
            logger.debug(f"Cache hit for {symbol} at {as_of_date}")
            return cached

        # Determine which sources to fetch
        sources = include_sources or set(DataSourceType)

        # Create result container
        result = AnalysisData(symbol=symbol, as_of_date=as_of_date)

        # Build list of sync fetch functions to execute in thread pool
        # Use functools.partial to avoid closure issues
        from functools import partial

        fetch_funcs = []

        if DataSourceType.INSIDER_SENTIMENT in sources:
            fetch_funcs.append(partial(self._fetch_insider_data_sync, symbol, as_of_date))

        if DataSourceType.INSTITUTIONAL_HOLDINGS in sources:
            fetch_funcs.append(partial(self._fetch_institutional_data_sync, symbol, as_of_date))

        if DataSourceType.SHORT_INTEREST in sources:
            fetch_funcs.append(partial(self._fetch_short_interest_sync, symbol, as_of_date))

        if DataSourceType.TREASURY_YIELDS in sources:
            fetch_funcs.append(partial(self._fetch_treasury_data_sync, as_of_date))

        if DataSourceType.MACRO_INDICATORS in sources:
            fetch_funcs.append(partial(self._fetch_macro_data_sync, as_of_date))

        if DataSourceType.MARKET_REGIME in sources:
            fetch_funcs.append(partial(self._fetch_market_regime_sync, as_of_date))

        if DataSourceType.CREDIT_RISK in sources:
            fetch_funcs.append(partial(self._fetch_credit_risk_sync, symbol, as_of_date))

        if DataSourceType.TECHNICAL_INDICATORS in sources:
            fetch_funcs.append(partial(self._fetch_technical_data_sync, symbol, as_of_date))

        if DataSourceType.PRICE_DATA in sources:
            fetch_funcs.append(partial(self._fetch_price_data_sync, symbol, as_of_date))

        if DataSourceType.REGIONAL_FED in sources:
            fetch_funcs.append(partial(self._fetch_regional_fed_data_sync, as_of_date))

        if DataSourceType.CBOE_VOLATILITY in sources:
            fetch_funcs.append(partial(self._fetch_cboe_data_sync, as_of_date))

        # Execute all fetches concurrently using thread pool
        if fetch_funcs:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(fetch_funcs), 5)) as executor:
                futures = [executor.submit(func) for func in fetch_funcs]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        fetch_result = future.result(timeout=30)
                        if fetch_result:
                            self._apply_result(result, fetch_result)
                    except Exception as e:
                        logger.warning(f"Data fetch error for {symbol}: {e}")

        # Cache result
        if symbol not in self._cache:
            self._cache[symbol] = {}
        self._cache[symbol][as_of_date] = result

        # Periodic cache cleanup
        self._maybe_clean_cache()

        return result

    async def get_historical_data(
        self,
        symbol: str,
        as_of_date: date,
        include_sources: Optional[Set[DataSourceType]] = None,
    ) -> AnalysisData:
        """Get analysis data as of a historical date (async wrapper).

        Wraps the sync version for async compatibility.
        """
        # Run sync version in thread to avoid blocking event loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.get_historical_data_sync, symbol, as_of_date, include_sources)
            return future.result(timeout=60)

    async def get_batch_data(
        self,
        symbols: List[str],
        as_of_date: Optional[date] = None,
        include_sources: Optional[Set[DataSourceType]] = None,
    ) -> Dict[str, AnalysisData]:
        """Get analysis data for multiple symbols efficiently.

        Args:
            symbols: List of stock ticker symbols.
            as_of_date: Date to get data as of (default: today).
            include_sources: Optional set of sources to include.

        Returns:
            Dict mapping symbol to AnalysisData.
        """
        as_of_date = as_of_date or date.today()

        # Fetch all symbols concurrently
        tasks = [self.get_historical_data(symbol, as_of_date, include_sources) for symbol in symbols]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dict
        result_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch data for {symbol}: {result}")
                result_dict[symbol] = AnalysisData(symbol=symbol, as_of_date=as_of_date)
            else:
                result_dict[symbol] = result

        return result_dict

    # Data source fetch methods

    def _fetch_insider_data_sync(self, symbol: str, as_of_date: date) -> Dict[str, Any]:
        """Fetch insider sentiment data from database (synchronous)."""
        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db = get_db_manager()
            with db.get_session() as session:
                result = session.execute(
                    text(
                        """
                        SELECT sentiment_score, buy_count, sell_count,
                               buy_value, sell_value, cluster_detected
                        FROM insider_sentiment
                        WHERE symbol = :symbol
                          AND calculation_date <= :as_of_date
                          AND period_days = :period_days
                        ORDER BY calculation_date DESC
                        LIMIT 1
                    """
                    ),
                    {
                        "symbol": symbol,
                        "as_of_date": as_of_date,
                        "period_days": INSIDER_PERIODS.standard_days,
                    },
                )
                row = result.fetchone()

                if row:
                    return {
                        "type": DataSourceType.INSIDER_SENTIMENT,
                        "data": {
                            "sentiment_score": float(row[0]) if row[0] else 0.0,
                            "buy_count": int(row[1]) if row[1] else 0,
                            "sell_count": int(row[2]) if row[2] else 0,
                            "buy_value": float(row[3]) if row[3] else 0.0,
                            "sell_value": float(row[4]) if row[4] else 0.0,
                            "cluster_detected": bool(row[5]) if row[5] else False,
                        },
                    }
        except Exception as e:
            logger.debug(f"Insider data fetch error for {symbol}: {e}")

        return {"type": DataSourceType.INSIDER_SENTIMENT, "data": {}}

    def _fetch_institutional_data_sync(self, symbol: str, as_of_date: date) -> Dict[str, Any]:
        """Fetch institutional holdings data from database (synchronous)."""
        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db = get_db_manager()
            with db.get_session() as session:
                # Join holdings with filings to get report dates
                result = session.execute(
                    text(
                        """
                        SELECT
                            SUM(h.shares) as total_shares,
                            COUNT(DISTINCT f.institution_id) as num_institutions,
                            SUM(h.value_thousands * 1000) as total_value
                        FROM form13f_holdings h
                        JOIN form13f_filings f ON h.filing_id = f.id
                        WHERE h.symbol = :symbol
                          AND f.report_quarter <= :as_of_date
                          AND f.report_quarter >= :as_of_date - INTERVAL '90 days'
                    """
                    ),
                    {"symbol": symbol, "as_of_date": as_of_date},
                )
                row = result.fetchone()

                if row and row[0]:
                    return {
                        "type": DataSourceType.INSTITUTIONAL_HOLDINGS,
                        "data": {
                            "total_institutional_shares": float(row[0]) if row[0] else 0,
                            "num_institutions": int(row[1]) if row[1] else 0,
                            "total_institutional_value": float(row[2]) if row[2] else 0,
                        },
                    }
        except Exception as e:
            logger.debug(f"Institutional data fetch error for {symbol}: {e}")

        return {"type": DataSourceType.INSTITUTIONAL_HOLDINGS, "data": {}}

    def _fetch_short_interest_sync(self, symbol: str, as_of_date: date) -> Dict[str, Any]:
        """Fetch short interest data from database (synchronous)."""
        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db = get_db_manager()
            with db.get_session() as session:
                result = session.execute(
                    text(
                        """
                        SELECT short_interest, avg_daily_volume,
                               days_to_cover, short_interest_ratio
                        FROM short_interest
                        WHERE symbol = :symbol
                          AND settlement_date <= :as_of_date
                        ORDER BY settlement_date DESC
                        LIMIT 1
                    """
                    ),
                    {"symbol": symbol, "as_of_date": as_of_date},
                )
                row = result.fetchone()

                if row:
                    return {
                        "type": DataSourceType.SHORT_INTEREST,
                        "data": {
                            "short_interest": float(row[0]) if row[0] else 0,
                            "avg_daily_volume": float(row[1]) if row[1] else 0,
                            "days_to_cover": float(row[2]) if row[2] else 0,
                            "short_percent_of_float": float(row[3]) if row[3] else 0,
                        },
                    }
        except Exception as e:
            logger.debug(f"Short interest fetch error for {symbol}: {e}")

        return {"type": DataSourceType.SHORT_INTEREST, "data": {}}

    def _fetch_treasury_data_sync(self, as_of_date: date) -> Dict[str, Any]:
        """Fetch treasury yield data from database (synchronous)."""
        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db = get_db_manager()
            with db.get_session() as session:
                result = session.execute(
                    text(
                        """
                        SELECT yield_1m, yield_3m, yield_6m, yield_1y,
                               yield_2y, yield_5y, yield_10y, yield_30y,
                               spread_10y_2y, spread_10y_3m, is_inverted
                        FROM treasury_yields
                        WHERE date <= :as_of_date
                        ORDER BY date DESC
                        LIMIT 1
                    """
                    ),
                    {"as_of_date": as_of_date},
                )
                row = result.fetchone()

                if row:
                    return {
                        "type": DataSourceType.TREASURY_YIELDS,
                        "data": {
                            "yield_1mo": float(row[0]) if row[0] else None,
                            "yield_3mo": float(row[1]) if row[1] else None,
                            "yield_6mo": float(row[2]) if row[2] else None,
                            "yield_1yr": float(row[3]) if row[3] else None,
                            "yield_2yr": float(row[4]) if row[4] else None,
                            "yield_5yr": float(row[5]) if row[5] else None,
                            "yield_10yr": float(row[6]) if row[6] else None,
                            "yield_30yr": float(row[7]) if row[7] else None,
                            "spread_10y_2y": float(row[8]) if row[8] else None,
                            "spread_10y_3mo": float(row[9]) if row[9] else None,
                            "is_inverted": bool(row[10]) if row[10] is not None else False,
                        },
                    }
        except Exception as e:
            logger.debug(f"Treasury data fetch error: {e}")

        return {"type": DataSourceType.TREASURY_YIELDS, "data": {}}

    def _fetch_macro_data_sync(self, as_of_date: date) -> Dict[str, Any]:
        """Fetch macro indicator data from database (synchronous)."""
        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db = get_db_manager()
            with db.get_session() as session:
                # Key macro indicators for RL features
                key_indicators = [
                    "VIXCLS",  # VIX volatility
                    "BAMLH0A0HYM2",  # High yield spread
                    "T10Y2Y",  # 10Y-2Y spread
                    "UNRATE",  # Unemployment rate
                    "CPIAUCSL",  # CPI inflation
                ]

                # Join with macro_indicators to get series_id
                result = session.execute(
                    text(
                        """
                        SELECT mi.series_id, mv.value
                        FROM macro_indicator_values mv
                        JOIN macro_indicators mi ON mv.indicator_id = mi.id
                        WHERE mi.series_id IN :indicators
                          AND mv.date <= :as_of_date
                        ORDER BY mi.series_id, mv.date DESC
                    """
                    ),
                    {"indicators": tuple(key_indicators), "as_of_date": as_of_date},
                )
                rows = result.fetchall()

                indicators = {}
                seen = set()
                for row in rows:
                    if row[0] not in seen:
                        indicators[row[0]] = float(row[1]) if row[1] else None
                        seen.add(row[0])

                return {
                    "type": DataSourceType.MACRO_INDICATORS,
                    "data": indicators,
                }
        except Exception as e:
            logger.debug(f"Macro data fetch error: {e}")

        return {"type": DataSourceType.MACRO_INDICATORS, "data": {}}

    def _fetch_market_regime_sync(self, as_of_date: date) -> Dict[str, Any]:
        """Fetch market regime classification from database (synchronous)."""
        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db = get_db_manager()
            with db.get_session() as session:
                result = session.execute(
                    text(
                        """
                        SELECT regime, credit_cycle_phase, volatility_regime,
                               recession_probability, yield_curve_inverted,
                               risk_off_signal
                        FROM market_regime_history
                        WHERE snapshot_date <= :as_of_date
                        ORDER BY snapshot_date DESC
                        LIMIT 1
                    """
                    ),
                    {"as_of_date": as_of_date},
                )
                row = result.fetchone()

                if row:
                    return {
                        "type": DataSourceType.MARKET_REGIME,
                        "data": {
                            "regime": row[0],
                            "credit_cycle_phase": row[1],
                            "volatility_regime": row[2],
                            "recession_probability": float(row[3]) if row[3] else 0,
                            "yield_curve_inverted": bool(row[4]) if row[4] is not None else False,
                            "risk_off_signal": bool(row[5]) if row[5] is not None else False,
                        },
                    }
        except Exception as e:
            logger.debug(f"Market regime fetch error: {e}")

        return {"type": DataSourceType.MARKET_REGIME, "data": {}}

    def _fetch_credit_risk_sync(self, symbol: str, as_of_date: date) -> Dict[str, Any]:
        """Fetch credit risk scores from database (synchronous)."""
        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db = get_db_manager()
            with db.get_session() as session:
                result = session.execute(
                    text(
                        """
                        SELECT altman_z_score, beneish_m_score,
                               piotroski_f_score, distress_tier
                        FROM credit_risk_scores
                        WHERE symbol = :symbol
                          AND calculation_date <= :as_of_date
                        ORDER BY calculation_date DESC
                        LIMIT 1
                    """
                    ),
                    {"symbol": symbol, "as_of_date": as_of_date},
                )
                row = result.fetchone()

                if row:
                    return {
                        "type": DataSourceType.CREDIT_RISK,
                        "data": {
                            "altman_z_score": float(row[0]) if row[0] else None,
                            "beneish_m_score": float(row[1]) if row[1] else None,
                            "piotroski_f_score": int(row[2]) if row[2] else None,
                            "distress_tier": row[3],
                        },
                    }
        except Exception as e:
            logger.debug(f"Credit risk fetch error for {symbol}: {e}")

        return {"type": DataSourceType.CREDIT_RISK, "data": {}}

    def _fetch_technical_data_sync(self, symbol: str, as_of_date: date) -> Dict[str, Any]:
        """Fetch technical indicator data (synchronous)."""
        # Technical data is typically calculated on-demand from price history
        # This would integrate with the technical analysis service
        return {"type": DataSourceType.TECHNICAL_INDICATORS, "data": {}}

    def _get_stock_engine(self):
        """Get or create the stock database engine (lazy initialization)."""
        if self._stock_engine is None:
            from sqlalchemy import create_engine

            from investigator.domain.services.market_data import get_stock_db_url

            self._stock_engine = create_engine(
                get_stock_db_url(),
                pool_size=3,
                max_overflow=5,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        return self._stock_engine

    def _fetch_price_data_sync(self, symbol: str, as_of_date: date) -> Dict[str, Any]:
        """Fetch price data from stock database (synchronous)."""
        try:
            from sqlalchemy import text

            # tickerdata is in the 'stock' database
            stock_engine = self._get_stock_engine()

            with stock_engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                        SELECT adjclose
                        FROM tickerdata
                        WHERE ticker = :symbol
                          AND date <= :as_of_date
                        ORDER BY date DESC
                        LIMIT 1
                    """
                    ),
                    {"symbol": symbol.upper(), "as_of_date": as_of_date},
                )
                row = result.fetchone()

                if row:
                    return {
                        "type": DataSourceType.PRICE_DATA,
                        "data": {
                            "current_price": float(row[0]) if row[0] else None,
                        },
                    }
        except Exception as e:
            logger.debug(f"Price data fetch error for {symbol}: {e}")

        return {"type": DataSourceType.PRICE_DATA, "data": {}}

    def _fetch_regional_fed_data_sync(self, as_of_date: date) -> Dict[str, Any]:
        """Fetch regional Federal Reserve indicators from database (synchronous).

        Returns data from all Fed districts including:
        - Atlanta Fed: GDPNow, Wage Growth Tracker, Business Inflation Expectations
        - Chicago Fed: CFNAI, NFCI, ANFCI
        - Cleveland Fed: Inflation Expectations, Median CPI, Trimmed Mean CPI
        - Dallas Fed: Texas Manufacturing, Trimmed Mean PCE
        - Kansas City Fed: KCFSI, LMCI, Manufacturing Survey
        - Philadelphia Fed: Manufacturing Survey, Leading Index
        - Richmond Fed: Manufacturing Survey
        - New York Fed: Recession Probability, Empire State, GSCPI
        """
        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db = get_db_manager()
            with db.get_session() as session:
                # Fetch all regional Fed indicators for the date
                result = session.execute(
                    text(
                        """
                        SELECT district, indicator_name, indicator_data, observation_date
                        FROM regional_fed_indicators
                        WHERE observation_date <= :as_of_date
                        ORDER BY district, indicator_name, observation_date DESC
                    """
                    ),
                    {"as_of_date": as_of_date},
                )
                rows = result.fetchall()

                # Build dict of latest value per district/indicator
                indicators = {}
                seen = set()
                for row in rows:
                    key = f"{row[0]}_{row[1]}"
                    if key not in seen:
                        district = row[0]
                        indicator = row[1]

                        if district not in indicators:
                            indicators[district] = {}

                        # Parse JSON data
                        data = row[2]
                        if isinstance(data, str):
                            import json

                            data = json.loads(data)

                        # Extract value
                        value = data.get("value") if isinstance(data, dict) else data
                        indicators[district][indicator] = {
                            "value": value,
                            "date": row[3].isoformat() if row[3] else None,
                        }
                        seen.add(key)

                # Also extract key summary metrics for RL features
                summary = {
                    # GDP outlook
                    "gdpnow": self._extract_indicator_value(indicators, "atlanta_fed", "gdpnow"),
                    # Financial conditions
                    "cfnai": self._extract_indicator_value(indicators, "chicago_fed", "cfnai"),
                    "nfci": self._extract_indicator_value(indicators, "chicago_fed", "nfci"),
                    "kcfsi": self._extract_indicator_value(indicators, "kansas_city_fed", "kcfsi"),
                    # Inflation
                    "inflation_expectations": self._extract_indicator_value(
                        indicators, "cleveland_fed", "inflation_expectations"
                    ),
                    "trimmed_mean_pce": self._extract_indicator_value(indicators, "dallas_fed", "trimmed_mean_pce"),
                    "median_cpi": self._extract_indicator_value(indicators, "cleveland_fed", "median_cpi"),
                    # Recession
                    "recession_probability": self._extract_indicator_value(
                        indicators, "new_york_fed", "recession_probability"
                    ),
                    # Manufacturing
                    "empire_state_mfg": self._extract_indicator_value(indicators, "new_york_fed", "empire_state_mfg"),
                }

                return {
                    "type": DataSourceType.REGIONAL_FED,
                    "data": {
                        "by_district": indicators,
                        "summary": summary,
                    },
                }
        except Exception as e:
            logger.debug(f"Regional Fed data fetch error: {e}")

        return {"type": DataSourceType.REGIONAL_FED, "data": {}}

    def _extract_indicator_value(
        self,
        indicators: Dict[str, Any],
        district: str,
        indicator: str,
    ) -> Optional[float]:
        """Extract a single indicator value from the indicators dict."""
        try:
            if district in indicators and indicator in indicators[district]:
                val = indicators[district][indicator].get("value")
                if val is not None:
                    return float(val)
        except (ValueError, TypeError):
            pass
        return None

    def _fetch_cboe_data_sync(self, as_of_date: date) -> Dict[str, Any]:
        """Fetch CBOE volatility data from database or cache (synchronous).

        Returns:
        - VIX: Current volatility level
        - VIX3M: 3-month volatility
        - SKEW: Tail risk indicator
        - Term structure: Contango/backwardation
        - Volatility regime classification
        """
        try:
            from sqlalchemy import text

            from investigator.infrastructure.database.db import get_db_manager

            db = get_db_manager()
            with db.get_session() as session:
                # First try regional_fed_indicators (where CBOE data is stored)
                result = session.execute(
                    text(
                        """
                        SELECT indicator_name, indicator_data, observation_date
                        FROM regional_fed_indicators
                        WHERE district = 'cboe'
                          AND observation_date <= :as_of_date
                        ORDER BY indicator_name, observation_date DESC
                    """
                    ),
                    {"as_of_date": as_of_date},
                )
                rows = result.fetchall()

                cboe_data = {}
                seen = set()
                for row in rows:
                    indicator = row[0]
                    if indicator not in seen:
                        data = row[1]
                        if isinstance(data, str):
                            import json

                            data = json.loads(data)

                        # Extract value
                        value = data.get("value") if isinstance(data, dict) else data
                        cboe_data[indicator] = {
                            "value": value,
                            "date": row[2].isoformat() if row[2] else None,
                        }
                        if isinstance(data, dict) and "ohlcv" in data:
                            cboe_data[indicator]["ohlcv"] = data["ohlcv"]
                        seen.add(indicator)

                # Extract key values for RL features
                vix = cboe_data.get("vix", {}).get("value")
                vix3m = cboe_data.get("vix3m", {}).get("value")
                skew = cboe_data.get("skew", {}).get("value")

                # Classify volatility regime
                volatility_regime = self._classify_volatility_regime(vix)

                # Detect term structure (contango if VIX3M > VIX)
                term_structure = None
                if vix and vix3m:
                    ratio = vix3m / vix
                    if ratio > 1.10:
                        term_structure = "steep_contango"
                    elif ratio > 1.02:
                        term_structure = "contango"
                    elif ratio > 0.98:
                        term_structure = "flat"
                    elif ratio > 0.90:
                        term_structure = "backwardation"
                    else:
                        term_structure = "steep_backwardation"

                return {
                    "type": DataSourceType.CBOE_VOLATILITY,
                    "data": {
                        "vix": vix,
                        "vix3m": vix3m,
                        "skew": skew,
                        "volatility_regime": volatility_regime,
                        "term_structure": term_structure,
                        "is_backwardation": term_structure in ("backwardation", "steep_backwardation"),
                        "skew_elevated": skew > 130 if skew else False,
                        "details": cboe_data,
                    },
                }
        except Exception as e:
            logger.debug(f"CBOE data fetch error: {e}")

        return {"type": DataSourceType.CBOE_VOLATILITY, "data": {}}

    def _classify_volatility_regime(self, vix: Optional[float]) -> str:
        """Classify volatility regime based on VIX level."""
        if vix is None:
            return "unknown"
        if vix < 12:
            return "very_low"
        elif vix < 15:
            return "low"
        elif vix < 20:
            return "normal"
        elif vix < 25:
            return "elevated"
        elif vix < 35:
            return "high"
        else:
            return "extreme"

    def _apply_result(self, analysis_data: AnalysisData, fetch_result: Dict[str, Any]):
        """Apply a fetch result to the AnalysisData object."""
        source_type = fetch_result.get("type")
        data = fetch_result.get("data", {})

        if source_type == DataSourceType.INSIDER_SENTIMENT:
            analysis_data.insider_data = data
        elif source_type == DataSourceType.INSTITUTIONAL_HOLDINGS:
            analysis_data.institutional_data = data
        elif source_type == DataSourceType.SHORT_INTEREST:
            analysis_data.short_interest = data
        elif source_type == DataSourceType.TREASURY_YIELDS:
            analysis_data.treasury_data = data
        elif source_type == DataSourceType.MACRO_INDICATORS:
            analysis_data.macro_indicators = data
        elif source_type == DataSourceType.MARKET_REGIME:
            analysis_data.market_regime = data
        elif source_type == DataSourceType.CREDIT_RISK:
            analysis_data.credit_risk = data
        elif source_type == DataSourceType.TECHNICAL_INDICATORS:
            analysis_data.technical_indicators = data
        elif source_type == DataSourceType.PRICE_DATA:
            if "current_price" in data:
                analysis_data.current_price = data["current_price"]
        elif source_type == DataSourceType.REGIONAL_FED:
            analysis_data.regional_fed_indicators = data
        elif source_type == DataSourceType.CBOE_VOLATILITY:
            analysis_data.cboe_data = data

    def _maybe_clean_cache(self):
        """Periodically clean old cache entries."""
        now = datetime.now()
        if now - self._last_cache_clean > self._cache_ttl:
            # Clean entries older than TTL
            cutoff = (now - self._cache_ttl).date()
            for symbol in list(self._cache.keys()):
                self._cache[symbol] = {d: v for d, v in self._cache[symbol].items() if d >= cutoff}
                if not self._cache[symbol]:
                    del self._cache[symbol]
            self._last_cache_clean = now


# Singleton instance
_facade_instance: Optional[DataSourceFacade] = None


def get_data_source_facade() -> DataSourceFacade:
    """Get or create singleton DataSourceFacade instance."""
    global _facade_instance
    if _facade_instance is None:
        _facade_instance = DataSourceFacade()
    return _facade_instance
