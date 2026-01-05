"""
Macro Indicators Module

Fetches and analyzes macroeconomic indicators from FRED (Federal Reserve Economic Data)
stored in the stock database. Includes calculations for derived indicators like the
Buffett Indicator (Stock Market to GDP ratio).

Database Tables:
- macro_indicators: Metadata about indicators (id, name, frequency, units)
- macro_indicator_values: Time series values with temporal validity tracking
"""

import logging
import os
import ssl
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

from contextlib import contextmanager

import aiohttp
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from investigator.config import get_config

# FRED API Configuration
FRED_API_BASE = "https://api.stlouisfed.org/fred"

logger = logging.getLogger(__name__)


def _get_fred_api_key() -> Optional[str]:
    """Get FRED API key from victor keyring or environment.

    Resolution order:
    1. Environment variable FRED_API_KEY (for CI/automation)
    2. Victor keyring (secure storage)
    3. None (will fail gracefully)
    """
    # Priority 1: Environment variable
    env_key = os.environ.get("FRED_API_KEY")
    if env_key:
        return env_key

    # Priority 2: Try victor keyring
    try:
        from victor.config.api_keys import get_service_key

        key = get_service_key("fred")
        if key:
            return key
    except ImportError:
        pass  # victor framework not available

    return None


def get_stock_db_manager():
    """
    Get database manager configured for stock database

    Returns a database session maker for the stock database where macro indicators
    and ticker price data are stored (separate from sec_database).
    """
    config = get_config()

    # Build connection URL for stock database
    db_config = config.database
    stock_db_url = f"postgresql://{db_config.username}:{db_config.password}@{db_config.host}:{db_config.port}/stock"

    # Create engine for stock database
    engine = create_engine(
        stock_db_url, pool_size=db_config.pool_size, max_overflow=db_config.max_overflow, echo=False, pool_pre_ping=True
    )

    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return SessionLocal


class MacroIndicatorsFetcher:
    """Fetches and processes macro indicators from FRED database"""

    # Key indicators for investment analysis
    KEY_INDICATORS = {
        # Economic Growth
        "GDP": "Gross Domestic Product",
        "GDPC1": "Real GDP",
        "A939RX0Q048SBEA": "Real GDP Growth Rate",
        "GDPNOW": "GDPNow Forecast",
        "NYGDPPCAPKDUSA": "GDP per Capita",
        # Equity Market
        "SP500": "S&P 500 Index",
        # Labor Market
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Total Nonfarm Payrolls",
        "JTSJOL": "Job Openings",
        # Inflation
        "CPIAUCSL": "Consumer Price Index",
        "PCEPI": "PCE Price Index",
        "CORESTICKM159SFRBATL": "Sticky Price CPI",
        "T10YIE": "10-Year Breakeven Inflation Rate",
        # Interest Rates & Credit
        "FEDFUNDS": "Federal Funds Rate",
        "DFF": "Fed Funds Effective Rate",
        "DGS10": "10-Year Treasury Rate",
        "T10Y2Y": "10Y-2Y Treasury Spread",
        "BAMLH0A0HYM2": "High Yield Credit Spread",
        "MORTGAGE30US": "30-Year Mortgage Rate",
        # Debt Metrics
        "GFDEGDQ188S": "Federal Debt to GDP",
        "GFDGDPA188S": "Gross Federal Debt to GDP",
        "HDTGPDUSQ163N": "Household Debt to GDP",
        "CMDEBT": "Household Credit Market Debt",
        "NCBDBIQ027S": "Corporate Debt Securities",
        "TBSDODNS": "Total Business Debt",
        "TDSP": "Household Debt Service Ratio",
        "FODSP": "Financial Obligations Ratio",
        # Sentiment & Risk
        "VIXCLS": "VIX Volatility Index",
        "UMCSENT": "Consumer Sentiment",
        # Housing
        "HOUST": "Housing Starts",
        "CSUSHPISA": "Case-Shiller Home Price Index",
        # Trade & Dollar
        "DTWEXBGS": "Trade Weighted Dollar Index",
        "BOPGSTB": "Trade Balance",
        "DEXUSEU": "USD/Euro Exchange Rate",
        # Monetary Policy
        "M2SL": "M2 Money Stock",
        "WALCL": "Fed Total Assets",
        "PSAVERT": "Personal Saving Rate",
        # Other
        "INDPRO": "Industrial Production",
        "RETAILIMSA": "Retail Sales",
        "PCE": "Personal Consumption Expenditures",
    }

    def __init__(self):
        """Initialize the macro indicators fetcher"""
        self.logger = logging.getLogger(__name__)
        self.SessionLocal = get_stock_db_manager()
        self._api_key = _get_fred_api_key()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=SSL_CONTEXT)
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), connector=connector)
        return self._session

    async def get_indicator_data(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """Fetch indicator data from FRED API.

        Args:
            series_id: FRED series ID (e.g., 'UNRATE', 'GDP')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dict with indicator metadata and values:
            {
                'name': str,
                'category': str,
                'frequency': str,
                'units': str,
                'values': [{'date': str, 'value': float}, ...]
            }
        """
        if not self._api_key:
            self.logger.warning("FRED_API_KEY not configured. " "Set via: victor keys --set-service fred --keyring")
            return {}

        result: Dict[str, Any] = {
            "name": self.KEY_INDICATORS.get(series_id, series_id),
            "category": "unknown",
            "frequency": "daily",
            "units": "",
            "values": [],
        }

        try:
            session = await self._get_session()

            # Fetch series metadata
            meta_url = f"{FRED_API_BASE}/series"
            meta_params = {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
            }

            async with session.get(meta_url, params=meta_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "seriess" in data and data["seriess"]:
                        series_info = data["seriess"][0]
                        result["name"] = series_info.get("title", result["name"])
                        result["frequency"] = series_info.get("frequency", "daily")
                        result["units"] = series_info.get("units", "")

            # Fetch observations
            obs_url = f"{FRED_API_BASE}/series/observations"
            obs_params = {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
                "sort_order": "desc",
            }

            async with session.get(obs_url, params=obs_params) as response:
                if response.status == 200:
                    data = await response.json()
                    observations = data.get("observations", [])

                    for obs in observations:
                        value = obs.get("value")
                        if value and value != ".":
                            try:
                                result["values"].append(
                                    {
                                        "date": obs.get("date"),
                                        "value": float(value),
                                    }
                                )
                            except ValueError:
                                pass

            self.logger.debug(f"Fetched {len(result['values'])} values for {series_id}")

        except Exception as e:
            self.logger.error(f"Error fetching {series_id} from FRED: {e}")

        return result

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    @contextmanager
    def get_session(self):
        """Get stock database session context manager"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def get_latest_values(
        self, indicator_ids: Optional[List[str]] = None, lookback_days: int = 1095
    ) -> Dict[str, Dict]:
        """
        Get latest values for specified indicators

        Args:
            indicator_ids: List of indicator IDs to fetch (None = all key indicators)
            lookback_days: How far back to look for data (default 1095 days = 3 years)

        Returns:
            Dict mapping indicator_id to {value, date, name, units, change_*}
        """
        if indicator_ids is None:
            indicator_ids = list(self.KEY_INDICATORS.keys())

        try:
            with self.get_session() as session:
                # Get latest value for each indicator
                # Build the query with dynamic interval
                lookback_interval = f"{lookback_days} days"
                query = text(
                    f"""
                    WITH latest_values AS (
                        SELECT DISTINCT ON (indicator_id)
                            indicator_id,
                            date,
                            value
                        FROM macro_indicator_values
                        WHERE indicator_id = ANY(:indicators)
                          AND is_current = true
                          AND date >= CURRENT_DATE - INTERVAL '{lookback_interval}'
                        ORDER BY indicator_id, date DESC
                    ),
                    previous_values AS (
                        SELECT DISTINCT ON (indicator_id)
                            indicator_id,
                            date AS prev_date,
                            value AS prev_value
                        FROM macro_indicator_values
                        WHERE indicator_id = ANY(:indicators)
                          AND is_current = true
                          AND date < (SELECT date FROM latest_values lv WHERE lv.indicator_id = macro_indicator_values.indicator_id)
                          AND date >= CURRENT_DATE - INTERVAL '{lookback_interval}'
                        ORDER BY indicator_id, date DESC
                    )
                    SELECT
                        lv.indicator_id,
                        lv.date,
                        lv.value,
                        pv.prev_date,
                        pv.prev_value,
                        mi.name,
                        mi.frequency,
                        mi.units
                    FROM latest_values lv
                    LEFT JOIN previous_values pv ON lv.indicator_id = pv.indicator_id
                    INNER JOIN macro_indicators mi ON lv.indicator_id = mi.id
                    ORDER BY lv.indicator_id
                """
                )

                result = session.execute(query, {"indicators": indicator_ids})

                indicators_data = {}
                for row in result:
                    indicator_id = row.indicator_id
                    value = float(row.value) if row.value else None
                    prev_value = float(row.prev_value) if row.prev_value else None

                    # Calculate changes
                    change_abs = value - prev_value if (value and prev_value) else None
                    change_pct = (
                        ((value - prev_value) / prev_value * 100)
                        if (value and prev_value and prev_value != 0)
                        else None
                    )

                    indicators_data[indicator_id] = {
                        "value": value,
                        "date": row.date,
                        "prev_value": prev_value,
                        "prev_date": row.prev_date if hasattr(row, "prev_date") else None,
                        "change_abs": change_abs,
                        "change_pct": change_pct,
                        "name": row.name,
                        "frequency": row.frequency,
                        "units": row.units,
                    }

                self.logger.info(f"Fetched {len(indicators_data)} macro indicators")
                return indicators_data

        except Exception as e:
            self.logger.error(f"Error fetching macro indicators: {e}")
            return {}

    def get_latest_indicators(
        self,
        indicator_ids: Optional[List[str]] = None,
        lookback_days: int = 1095,
    ) -> Dict[str, Dict]:
        """
        Backwards-compatible alias for legacy callers expecting get_latest_indicators().
        """
        self.logger.debug(
            "get_latest_indicators() is deprecated; use get_latest_values() instead. "
            "Maintaining compatibility for legacy valuation workflows."
        )
        return self.get_latest_values(indicator_ids=indicator_ids, lookback_days=lookback_days)

    def get_time_series(
        self,
        indicator_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Get time series data for an indicator

        Args:
            indicator_id: FRED series ID
            start_date: Start date for data (None = earliest available)
            end_date: End date for data (None = latest available)
            limit: Maximum number of data points

        Returns:
            DataFrame with columns: date, value
        """
        try:
            with self.get_session() as session:
                query = text(
                    """
                    SELECT date, value
                    FROM macro_indicator_values
                    WHERE indicator_id = :indicator_id
                      AND is_current = true
                      AND (:start_date IS NULL OR date >= :start_date)
                      AND (:end_date IS NULL OR date <= :end_date)
                    ORDER BY date DESC
                    LIMIT :limit
                """
                )

                result = session.execute(
                    query,
                    {"indicator_id": indicator_id, "start_date": start_date, "end_date": end_date, "limit": limit},
                )

                data = [(row.date, float(row.value)) for row in result if row.value]
                df = pd.DataFrame(data, columns=["date", "value"])
                df = df.sort_values("date")  # Sort ascending for analysis

                self.logger.debug(f"Fetched {len(df)} data points for {indicator_id}")
                return df

        except Exception as e:
            self.logger.error(f"Error fetching time series for {indicator_id}: {e}")
            return pd.DataFrame(columns=["date", "value"])

    def get_vti_price(self) -> Optional[float]:
        """
        Get latest VTI (Total Stock Market ETF) price from tickerdata table

        Returns:
            Latest VTI closing price or None
        """
        try:
            with self.get_session() as session:
                query = text(
                    """
                    SELECT close, date
                    FROM tickerdata
                    WHERE ticker = 'VTI'
                    ORDER BY date DESC
                    LIMIT 1
                """
                )
                result = session.execute(query)
                row = result.first()

                if row:
                    return {"price": float(row.close), "date": row.date}
                return None
        except Exception as e:
            self.logger.error(f"Error fetching VTI price: {e}")
            return None

    def calculate_buffett_indicator(self) -> Optional[Dict]:
        """
        Calculate the Buffett Indicator: Total Stock Market Cap / GDP

        Uses VTI (Vanguard Total Stock Market ETF) as proxy for Wilshire 5000 Total Market Index.

        Method:
        1. Get VTI price from tickerdata
        2. Get GDP from macro_indicators
        3. Calculate total market cap using VTI price and established multiplier
        4. Compute ratio: Total Market Cap / GDP * 100

        Multiplier Derivation (as of Sep 2025):
        - Wilshire 5000 market cap: ~$67 trillion
        - VTI total net assets: ~$1.91 trillion (all share classes)
        - Multiplier: $67T / $1.91T ≈ 35x

        However, VTI price represents price per share, not AUM.
        Better approach: Wilshire 5000 Full Cap Index ≈ VTI price * 200
        (Historical relationship: W5K at 66,848 when VTI around $334)

        Returns:
            Dict with indicator value, interpretation, and component values
        """
        try:
            indicators = self.get_latest_values(["GDP"])
            vti_data = self.get_vti_price()

            if "GDP" not in indicators:
                self.logger.warning("Cannot calculate Buffett Indicator: Missing GDP data")
                return None

            if not vti_data:
                self.logger.warning("Cannot calculate Buffett Indicator: Missing VTI price data")
                return None

            vti_price = vti_data["price"]
            gdp_value = indicators["GDP"]["value"]  # In billions

            if not vti_price or not gdp_value:
                return None

            # Estimate Wilshire 5000 index from VTI price
            # Historical correlation: W5K ≈ VTI * 200
            # Then convert to market cap: W5K index ≈ market cap in billions
            # (1 index point ≈ $1 billion market cap)
            estimated_w5k_index = vti_price * 200
            estimated_market_cap = estimated_w5k_index  # In billions

            # Calculate ratio (expressed as percentage)
            buffett_ratio = (estimated_market_cap / gdp_value) * 100

            # Interpretation ranges (Warren Buffett's guidance)
            # Source: Multiple analyses suggest these thresholds
            if buffett_ratio < 75:
                interpretation = "Significantly Undervalued"
                signal = "strong_buy"
            elif buffett_ratio < 90:
                interpretation = "Moderately Undervalued"
                signal = "buy"
            elif buffett_ratio < 115:
                interpretation = "Fair Value"
                signal = "neutral"
            elif buffett_ratio < 140:
                interpretation = "Moderately Overvalued"
                signal = "caution"
            else:
                interpretation = "Significantly Overvalued"
                signal = "warning"

            return {
                "ratio": buffett_ratio,
                "interpretation": interpretation,
                "signal": signal,
                "vti_price": vti_price,
                "vti_date": vti_data["date"],
                "gdp": gdp_value,
                "gdp_date": indicators["GDP"]["date"],
                "estimated_w5k_index": estimated_w5k_index,
                "estimated_market_cap": estimated_market_cap,
                "note": "Calculated using VTI ETF as proxy for Wilshire 5000 Total Market Index",
            }

        except Exception as e:
            self.logger.error(f"Error calculating Buffett Indicator: {e}")
            return None

    def get_macro_summary(self) -> Dict:
        """
        Get a comprehensive summary of macro conditions

        Returns:
            Dict with categorized indicators and overall assessment
        """
        indicators = self.get_latest_values()

        summary = {
            "timestamp": datetime.now().isoformat(),
            "indicators": indicators,
            "categories": {},
            "alerts": [],
            "overall_assessment": None,
        }

        # Categorize indicators
        categories = {
            "growth": ["GDP", "GDPC1", "A939RX0Q048SBEA", "GDPNOW", "NYGDPPCAPKDUSA"],
            "employment": ["UNRATE", "PAYEMS", "JTSJOL"],
            "inflation": ["CPIAUCSL", "PCEPI", "CORESTICKM159SFRBATL", "T10YIE"],
            "rates": ["FEDFUNDS", "DFF", "DGS10", "T10Y2Y", "MORTGAGE30US"],
            "credit": ["BAMLH0A0HYM2"],
            "debt": [
                "GFDEGDQ188S",
                "GFDGDPA188S",
                "HDTGPDUSQ163N",
                "CMDEBT",
                "NCBDBIQ027S",
                "TBSDODNS",
                "TDSP",
                "FODSP",
            ],
            "market": ["SP500", "VIXCLS"],
            "sentiment": ["UMCSENT"],
            "housing": ["HOUST", "CSUSHPISA"],
            "monetary": ["M2SL", "WALCL", "PSAVERT"],
            "trade": ["DTWEXBGS", "BOPGSTB", "DEXUSEU"],
        }

        for category, indicator_list in categories.items():
            summary["categories"][category] = {
                ind_id: indicators.get(ind_id) for ind_id in indicator_list if ind_id in indicators
            }

        # Generate alerts for significant changes or levels
        for ind_id, data in indicators.items():
            if data.get("change_pct"):
                # Alert on large changes
                if abs(data["change_pct"]) > 10:
                    summary["alerts"].append(
                        {
                            "type": "large_change",
                            "indicator": data["name"],
                            "change_pct": data["change_pct"],
                            "severity": "high" if abs(data["change_pct"]) > 20 else "medium",
                        }
                    )

        # Special indicators
        buffett = self.calculate_buffett_indicator()
        if buffett:
            summary["buffett_indicator"] = buffett

            if buffett["signal"] in ["warning", "strong_buy"]:
                summary["alerts"].append(
                    {
                        "type": "buffett_indicator",
                        "indicator": "Market Valuation",
                        "interpretation": buffett["interpretation"],
                        "ratio": buffett["ratio"],
                        "severity": "high" if buffett["signal"] == "warning" else "medium",
                    }
                )

        # Overall assessment
        risk_factors = len([a for a in summary["alerts"] if a["severity"] == "high"])
        if risk_factors == 0:
            summary["overall_assessment"] = "favorable"
        elif risk_factors <= 2:
            summary["overall_assessment"] = "mixed"
        else:
            summary["overall_assessment"] = "cautionary"

        return summary


def format_indicator_for_display(indicator_id: str, data: Dict) -> str:
    """
    Format an indicator for display in reports

    Args:
        indicator_id: FRED series ID
        data: Indicator data dict from get_latest_values()

    Returns:
        Formatted string for display
    """
    if not data or not data.get("value"):
        return f"{indicator_id}: No data available"

    value = data["value"]
    units = data.get("units", "")
    change_pct = data.get("change_pct")
    date = data.get("date", "unknown date")

    # Format value based on units
    if "Percent" in units or indicator_id in ["UNRATE", "FEDFUNDS", "DGS10"]:
        value_str = f"{value:.2f}%"
    elif "Index" in units:
        value_str = f"{value:,.2f}"
    elif "Billions" in units:
        value_str = f"${value:,.1f}B"
    elif "Millions" in units:
        value_str = f"${value:,.1f}M"
    elif "Thousands" in units:
        value_str = f"{value:,.0f}K"
    else:
        value_str = f"{value:,.2f}"

    # Add change if available
    if change_pct:
        arrow = "↑" if change_pct > 0 else "↓"
        change_str = f" {arrow} {abs(change_pct):.2f}%"
    else:
        change_str = ""

    return f"{data['name']}: {value_str}{change_str} (as of {date})"


# Singleton instance for scheduled collectors
_macro_indicator_fetcher: Optional[MacroIndicatorsFetcher] = None


def get_macro_indicator_fetcher() -> MacroIndicatorsFetcher:
    """Get singleton MacroIndicatorsFetcher instance for scheduled jobs.

    Returns:
        MacroIndicatorsFetcher instance
    """
    global _macro_indicator_fetcher
    if _macro_indicator_fetcher is None:
        _macro_indicator_fetcher = MacroIndicatorsFetcher()
    return _macro_indicator_fetcher


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    fetcher = MacroIndicatorsFetcher()

    print("\n=== Testing Macro Indicators Fetcher ===\n")

    # Test latest values
    print("Latest Values:")
    indicators = fetcher.get_latest_values(["GDP", "SP500", "UNRATE", "CPIAUCSL"])
    for ind_id, data in indicators.items():
        print(f"  {format_indicator_for_display(ind_id, data)}")

    # Test Buffett Indicator
    print("\nBuffett Indicator:")
    buffett = fetcher.calculate_buffett_indicator()
    if buffett:
        print(f"  Ratio: {buffett['ratio']:.1f}%")
        print(f"  Interpretation: {buffett['interpretation']}")
        print(f"  Signal: {buffett['signal']}")

    # Test macro summary
    print("\nMacro Summary:")
    summary = fetcher.get_macro_summary()
    print(f"  Overall Assessment: {summary['overall_assessment']}")
    print(f"  Number of Alerts: {len(summary['alerts'])}")
    for alert in summary["alerts"]:
        print(f"    - {alert['type']}: {alert.get('indicator', 'N/A')}")
