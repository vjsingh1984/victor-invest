"""
Treasury Yield Data Source

Provides Treasury yield curve data with spread calculations
and curve analysis (inversion detection, steepening, etc.)
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional
import logging

from ..base import (
    DataSource, DataResult, SourceMetadata,
    DataCategory, DataFrequency, DataQuality
)
from ..registry import register_source

logger = logging.getLogger(__name__)


TREASURY_MATURITIES = [
    ("1M", "DGS1MO"),
    ("3M", "DGS3MO"),
    ("6M", "DGS6MO"),
    ("1Y", "DGS1"),
    ("2Y", "DGS2"),
    ("3Y", "DGS3"),
    ("5Y", "DGS5"),
    ("7Y", "DGS7"),
    ("10Y", "DGS10"),
    ("20Y", "DGS20"),
    ("30Y", "DGS30"),
]


@register_source("treasury_yields", DataCategory.FIXED_INCOME)
class TreasuryYieldSource(DataSource):
    """
    Treasury Yield Curve Data Source

    Provides:
    - Full yield curve (1M to 30Y)
    - Key spreads (10Y-2Y, 10Y-3M)
    - Curve analysis (inversion, steepening)
    - Historical comparison
    """

    def __init__(self):
        super().__init__("treasury_yields", DataCategory.FIXED_INCOME, DataFrequency.DAILY)

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="treasury_yields",
            category=DataCategory.FIXED_INCOME,
            frequency=DataFrequency.DAILY,
            description="US Treasury yield curve with spread analysis",
            provider="US Treasury / FRED",
            is_free=True,
            requires_api_key=False,
            lookback_days=365 * 30,
            symbols_supported=False,
        )

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Fetch Treasury yields and calculate spreads"""
        try:
            from investigator.infrastructure.database.db import get_db_manager
            from sqlalchemy import text

            engine = get_db_manager().engine
            target_date = as_of_date or date.today()

            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT date, yield_1m, yield_3m, yield_6m, yield_1y,
                               yield_2y, yield_5y, yield_10y, yield_20y, yield_30y,
                               spread_10y_2y, spread_10y_3m, is_inverted
                        FROM treasury_yields
                        WHERE date <= :target_date
                        ORDER BY date DESC
                        LIMIT 1
                    """),
                    {"target_date": target_date}
                )
                row = result.fetchone()

            if not row:
                return DataResult(
                    success=False,
                    error="No Treasury data available",
                    source=self.name,
                )

            # Map columns to maturity labels
            yields = {
                "1M": {"rate": float(row[1]) if row[1] else None, "date": row[0].isoformat()},
                "3M": {"rate": float(row[2]) if row[2] else None, "date": row[0].isoformat()},
                "6M": {"rate": float(row[3]) if row[3] else None, "date": row[0].isoformat()},
                "1Y": {"rate": float(row[4]) if row[4] else None, "date": row[0].isoformat()},
                "2Y": {"rate": float(row[5]) if row[5] else None, "date": row[0].isoformat()},
                "5Y": {"rate": float(row[6]) if row[6] else None, "date": row[0].isoformat()},
                "10Y": {"rate": float(row[7]) if row[7] else None, "date": row[0].isoformat()},
                "20Y": {"rate": float(row[8]) if row[8] else None, "date": row[0].isoformat()},
                "30Y": {"rate": float(row[9]) if row[9] else None, "date": row[0].isoformat()},
            }

            # Filter out None rates
            yields = {k: v for k, v in yields.items() if v["rate"] is not None}

            # Use pre-calculated spreads from DB
            analysis = {
                "spread_10y_2y": float(row[10]) if row[10] else None,
                "spread_10y_3m": float(row[11]) if row[11] else None,
                "is_inverted_10_2": bool(row[12]) if row[12] is not None else None,
            }

            # Add additional analysis
            analysis.update(self._analyze_curve(yields))

            data = {
                "yields": yields,
                "as_of_date": row[0].isoformat(),
                **analysis,
            }

            return DataResult(
                success=True,
                data=data,
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            logger.error(f"Treasury fetch error: {e}")
            return DataResult(success=False, error=str(e), source=self.name)

    def _analyze_curve(self, yields: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze yield curve shape and spreads"""
        analysis = {}

        # Key spreads
        if "10Y" in yields and "2Y" in yields:
            spread_10_2 = yields["10Y"]["rate"] - yields["2Y"]["rate"]
            analysis["spread_10y_2y"] = round(spread_10_2, 3)
            analysis["is_inverted_10_2"] = spread_10_2 < 0

        if "10Y" in yields and "3M" in yields:
            spread_10_3m = yields["10Y"]["rate"] - yields["3M"]["rate"]
            analysis["spread_10y_3m"] = round(spread_10_3m, 3)
            analysis["is_inverted_10_3m"] = spread_10_3m < 0

        if "30Y" in yields and "10Y" in yields:
            spread_30_10 = yields["30Y"]["rate"] - yields["10Y"]["rate"]
            analysis["spread_30y_10y"] = round(spread_30_10, 3)

        # Curve shape classification
        if "is_inverted_10_2" in analysis and "is_inverted_10_3m" in analysis:
            if analysis["is_inverted_10_2"] and analysis["is_inverted_10_3m"]:
                analysis["curve_shape"] = "deeply_inverted"
                analysis["recession_signal"] = "strong"
            elif analysis["is_inverted_10_2"] or analysis["is_inverted_10_3m"]:
                analysis["curve_shape"] = "partially_inverted"
                analysis["recession_signal"] = "moderate"
            elif analysis.get("spread_10y_2y", 0) > 1.0:
                analysis["curve_shape"] = "steep"
                analysis["recession_signal"] = "low"
            else:
                analysis["curve_shape"] = "flat"
                analysis["recession_signal"] = "elevated"

        # Term premium estimate (simplified)
        if "10Y" in yields and "2Y" in yields:
            term_premium = (yields["10Y"]["rate"] - yields["2Y"]["rate"]) / 2
            analysis["term_premium_estimate"] = round(term_premium, 3)

        return analysis

    def get_historical_spreads(
        self,
        spread_type: str = "10y_2y",
        lookback_days: int = 365
    ) -> DataResult:
        """Get historical spread data"""
        try:
            from investigator.infrastructure.database.db import get_db_manager
            from sqlalchemy import text
            from datetime import timedelta

            engine = get_db_manager().engine
            start_date = date.today() - timedelta(days=lookback_days)

            if spread_type == "10y_2y":
                series_a, series_b = "DGS10", "DGS2"
            elif spread_type == "10y_3m":
                series_a, series_b = "DGS10", "DGS3MO"
            else:
                return DataResult(
                    success=False,
                    error=f"Unknown spread type: {spread_type}",
                    source=self.name,
                )

            # Map spread type to column
            spread_column = "spread_10y_2y" if spread_type == "10y_2y" else "spread_10y_3m"

            with engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        SELECT date, {spread_column} as spread
                        FROM treasury_yields
                        WHERE date >= :start_date
                        AND {spread_column} IS NOT NULL
                        ORDER BY date
                    """),
                    {"start_date": start_date}
                )
                rows = result.fetchall()

            if not rows:
                return DataResult(
                    success=False,
                    error="No historical data",
                    source=self.name,
                )

            spreads = [{"date": r[0].isoformat(), "spread": float(r[1])} for r in rows]

            # Calculate statistics
            values = [r[1] for r in rows]
            stats = {
                "current": values[-1] if values else None,
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "inversion_days": sum(1 for v in values if v < 0),
                "inversion_pct": sum(1 for v in values if v < 0) / len(values) * 100,
            }

            return DataResult(
                success=True,
                data={"spreads": spreads, "statistics": stats},
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)
