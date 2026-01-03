"""
FRED (Federal Reserve Economic Data) Source

Provides access to 800,000+ economic time series from the St. Louis Fed.
Free API with 120 requests/minute limit.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional
import logging

from ..base import (
    MacroDataSource, DataResult, SourceMetadata,
    DataCategory, DataFrequency, DataQuality
)
from ..registry import register_source

logger = logging.getLogger(__name__)


# Key FRED series organized by category
FRED_SERIES = {
    # Economic Activity
    "gdp": {
        "GDP": "Gross Domestic Product",
        "GDPC1": "Real GDP",
        "PAYEMS": "Total Nonfarm Payrolls",
        "UNRATE": "Unemployment Rate",
        "ICSA": "Initial Jobless Claims",
        "INDPRO": "Industrial Production Index",
    },
    # Inflation
    "inflation": {
        "CPIAUCSL": "CPI All Items",
        "CPILFESL": "Core CPI",
        "PCEPI": "PCE Price Index",
        "PCEPILFE": "Core PCE",
        "T5YIE": "5-Year Breakeven Inflation",
        "T10YIE": "10-Year Breakeven Inflation",
    },
    # Interest Rates
    "rates": {
        "DFF": "Fed Funds Rate",
        "DGS1MO": "1-Month Treasury",
        "DGS3MO": "3-Month Treasury",
        "DGS1": "1-Year Treasury",
        "DGS2": "2-Year Treasury",
        "DGS5": "5-Year Treasury",
        "DGS10": "10-Year Treasury",
        "DGS30": "30-Year Treasury",
    },
    # Spreads
    "spreads": {
        "T10Y2Y": "10Y-2Y Spread",
        "T10Y3M": "10Y-3M Spread",
        "BAMLH0A0HYM2": "High Yield Spread",
        "BAMLC0A4CBBB": "BBB Spread",
        "TEDRATE": "TED Spread",
    },
    # Volatility
    "volatility": {
        "VIXCLS": "VIX Close",
    },
    # Housing
    "housing": {
        "HOUST": "Housing Starts",
        "PERMIT": "Building Permits",
        "HSN1F": "New Home Sales",
        "CSUSHPINSA": "Case-Shiller Home Price Index",
    },
    # Consumer
    "consumer": {
        "UMCSENT": "Consumer Sentiment (Michigan)",
        "PCE": "Personal Consumption Expenditures",
        "RSAFS": "Retail Sales",
    },
    # Manufacturing
    "manufacturing": {
        "MANEMP": "Manufacturing Employment",
        "DGORDER": "Durable Goods Orders",
        "NEWORDER": "Manufacturers New Orders",
    },
}


@register_source("fred_macro", DataCategory.MACRO)
class FredMacroSource(MacroDataSource):
    """
    FRED Macro Economic Data Source

    Free API: https://fred.stlouisfed.org/docs/api/fred/
    Rate Limit: 120 requests/minute
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("fred_macro", DataFrequency.DAILY)
        self._api_key = api_key
        self._base_url = "https://api.stlouisfed.org/fred"

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="fred_macro",
            category=DataCategory.MACRO,
            frequency=DataFrequency.DAILY,
            description="Federal Reserve Economic Data - 800,000+ economic time series",
            provider="Federal Reserve Bank of St. Louis",
            is_free=True,
            requires_api_key=True,
            rate_limit_per_minute=120,
            lookback_days=365 * 50,  # 50 years of history
            symbols_supported=False,
        )

    def _get_api_key(self) -> str:
        """Get API key from config or keyring"""
        if self._api_key:
            return self._api_key

        try:
            from victor.config.api_keys import get_secret
            return get_secret("fred")
        except Exception:
            import os
            return os.environ.get("FRED_API_KEY", "")

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Fetch all FRED series"""
        try:
            from investigator.infrastructure.database.db import get_db_manager
            from sqlalchemy import text

            engine = get_db_manager().engine
            data = {}

            # Fetch from database (already collected by scheduler)
            with engine.connect() as conn:
                for category, series_dict in FRED_SERIES.items():
                    category_data = {}
                    for series_id, description in series_dict.items():
                        result = conn.execute(
                            text("""
                                SELECT v.value, v.date
                                FROM macro_indicator_values v
                                JOIN macro_indicators i ON v.indicator_id = i.id
                                WHERE i.series_id = :series_id
                                ORDER BY v.date DESC
                                LIMIT 1
                            """),
                            {"series_id": series_id}
                        )
                        row = result.fetchone()
                        if row:
                            category_data[series_id] = {
                                "value": float(row[0]),
                                "date": row[1].isoformat() if row[1] else None,
                                "description": description,
                            }

                    if category_data:
                        data[category] = category_data

            if not data:
                return DataResult(
                    success=False,
                    error="No FRED data found in database",
                    source=self.name,
                )

            # Calculate derived metrics
            data["derived"] = self._calculate_derived_metrics(data)

            return DataResult(
                success=True,
                data=data,
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            logger.error(f"FRED fetch error: {e}")
            return DataResult(
                success=False,
                error=str(e),
                source=self.name,
            )

    def _calculate_derived_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived metrics from raw data"""
        derived = {}

        # Yield curve inversion
        rates = data.get("rates", {})
        if "DGS10" in rates and "DGS2" in rates:
            spread_10_2 = rates["DGS10"]["value"] - rates["DGS2"]["value"]
            derived["yield_curve_inverted"] = spread_10_2 < 0
            derived["spread_10y_2y"] = spread_10_2

        if "DGS10" in rates and "DGS3MO" in rates:
            spread_10_3m = rates["DGS10"]["value"] - rates["DGS3MO"]["value"]
            derived["spread_10y_3m"] = spread_10_3m

        # Real rates
        inflation = data.get("inflation", {})
        if "DGS10" in rates and "T10YIE" in inflation:
            derived["real_rate_10y"] = rates["DGS10"]["value"] - inflation["T10YIE"]["value"]

        return derived

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> DataResult:
        """Fetch a specific FRED series"""
        try:
            from investigator.infrastructure.database.db import get_db_manager
            from sqlalchemy import text

            engine = get_db_manager().engine

            query = """
                SELECT v.value, v.date
                FROM macro_indicator_values v
                JOIN macro_indicators i ON v.indicator_id = i.id
                WHERE i.series_id = :series_id
            """
            params = {"series_id": series_id}

            if start_date:
                query += " AND v.date >= :start_date"
                params["start_date"] = start_date
            if end_date:
                query += " AND v.date <= :end_date"
                params["end_date"] = end_date

            query += " ORDER BY v.date"

            with engine.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()

            if not rows:
                return DataResult(
                    success=False,
                    error=f"No data for series: {series_id}",
                    source=self.name,
                )

            data = {
                "series_id": series_id,
                "values": [{"date": r[1].isoformat(), "value": float(r[0])} for r in rows],
                "count": len(rows),
            }

            return DataResult(
                success=True,
                data=data,
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            return DataResult(
                success=False,
                error=str(e),
                source=self.name,
            )

    def get_available_series(self) -> Dict[str, Dict[str, str]]:
        """Return available FRED series by category"""
        return FRED_SERIES
