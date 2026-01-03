"""
CBOE Volatility Data Source

Provides VIX, SKEW, term structure analysis and volatility regime classification.
Free data from CBOE and FRED.
"""

from datetime import date, datetime
from typing import Any, Dict, Optional
import logging

from ..base import (
    DataSource, DataResult, SourceMetadata,
    DataCategory, DataFrequency, DataQuality
)
from ..registry import register_source

logger = logging.getLogger(__name__)


@register_source("cboe_volatility", DataCategory.VOLATILITY)
class CBOEVolatilitySource(DataSource):
    """
    CBOE Volatility Data Source

    Provides:
    - VIX (spot and term structure)
    - SKEW Index
    - Volatility regime classification
    - Fear/greed indicators
    """

    def __init__(self):
        super().__init__("cboe_volatility", DataCategory.VOLATILITY, DataFrequency.DAILY)

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="cboe_volatility",
            category=DataCategory.VOLATILITY,
            frequency=DataFrequency.DAILY,
            description="CBOE VIX, SKEW, and volatility term structure",
            provider="CBOE / FRED",
            is_free=True,
            requires_api_key=False,
            lookback_days=365 * 20,
            symbols_supported=False,
        )

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Fetch CBOE volatility data"""
        try:
            from investigator.infrastructure.database.db import get_db_manager
            from sqlalchemy import text

            engine = get_db_manager().engine
            target_date = as_of_date or date.today()

            data = {}

            with engine.connect() as conn:
                # Fetch VIX from macro_indicator_values
                result = conn.execute(
                    text("""
                        SELECT v.value, v.date
                        FROM macro_indicator_values v
                        JOIN macro_indicators i ON v.indicator_id = i.id
                        WHERE i.series_id = 'VIXCLS'
                        AND v.date <= :target_date
                        ORDER BY v.date DESC
                        LIMIT 1
                    """),
                    {"target_date": target_date}
                )
                row = result.fetchone()
                if row:
                    data["vix"] = float(row[0])
                    data["vix_date"] = row[1].isoformat()

                # Fetch from regional_fed_indicators for CBOE data
                result = conn.execute(
                    text("""
                        SELECT indicator_name,
                               (indicator_data->>'value')::float as value,
                               observation_date
                        FROM regional_fed_indicators
                        WHERE district = 'cboe'
                        AND observation_date <= :target_date
                        ORDER BY observation_date DESC
                    """),
                    {"target_date": target_date}
                )
                for row in result:
                    indicator = row[0].lower()
                    if indicator == "vix" and "vix" not in data:
                        data["vix"] = float(row[1])
                    elif indicator == "vix3m":
                        data["vix3m"] = float(row[1])
                    elif indicator == "skew":
                        data["skew"] = float(row[1])

            if not data.get("vix"):
                return DataResult(
                    success=False,
                    error="No VIX data available",
                    source=self.name,
                )

            # Add analysis
            data.update(self._analyze_volatility(data))

            return DataResult(
                success=True,
                data=data,
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            logger.error(f"CBOE fetch error: {e}")
            return DataResult(success=False, error=str(e), source=self.name)

    def _analyze_volatility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility metrics"""
        analysis = {}
        vix = data.get("vix", 0)
        vix3m = data.get("vix3m")
        skew = data.get("skew")

        # VIX regime classification
        if vix < 12:
            analysis["volatility_regime"] = "extremely_low"
            analysis["regime_signal"] = "complacency_warning"
        elif vix < 16:
            analysis["volatility_regime"] = "low"
            analysis["regime_signal"] = "bullish"
        elif vix < 20:
            analysis["volatility_regime"] = "normal"
            analysis["regime_signal"] = "neutral"
        elif vix < 25:
            analysis["volatility_regime"] = "elevated"
            analysis["regime_signal"] = "cautious"
        elif vix < 30:
            analysis["volatility_regime"] = "high"
            analysis["regime_signal"] = "fear"
        else:
            analysis["volatility_regime"] = "extreme"
            analysis["regime_signal"] = "panic"

        # Term structure analysis
        if vix3m:
            term_structure = vix3m / vix
            analysis["term_structure_ratio"] = round(term_structure, 3)

            if term_structure < 0.9:
                analysis["term_structure"] = "steep_backwardation"
                analysis["is_backwardation"] = True
                analysis["fear_signal"] = "extreme"
            elif term_structure < 1.0:
                analysis["term_structure"] = "backwardation"
                analysis["is_backwardation"] = True
                analysis["fear_signal"] = "elevated"
            elif term_structure < 1.05:
                analysis["term_structure"] = "flat"
                analysis["is_backwardation"] = False
                analysis["fear_signal"] = "neutral"
            else:
                analysis["term_structure"] = "contango"
                analysis["is_backwardation"] = False
                analysis["fear_signal"] = "low"

        # SKEW analysis (tail risk)
        if skew:
            if skew > 150:
                analysis["skew_signal"] = "extreme_tail_risk"
            elif skew > 140:
                analysis["skew_signal"] = "elevated_tail_risk"
            elif skew > 130:
                analysis["skew_signal"] = "moderate_tail_risk"
            elif skew > 120:
                analysis["skew_signal"] = "normal"
            else:
                analysis["skew_signal"] = "low_tail_hedging"

        # Combined fear/greed score (0-100)
        fear_score = 50  # Start neutral

        # VIX contribution (0-40 points)
        if vix < 15:
            fear_score -= 20
        elif vix < 20:
            fear_score -= 10
        elif vix > 30:
            fear_score += 30
        elif vix > 25:
            fear_score += 20
        elif vix > 20:
            fear_score += 10

        # Term structure contribution (0-30 points)
        if analysis.get("is_backwardation"):
            fear_score += 25
        elif analysis.get("term_structure_ratio", 1) > 1.1:
            fear_score -= 15

        # SKEW contribution (0-20 points)
        if skew and skew > 140:
            fear_score += 15
        elif skew and skew < 120:
            fear_score -= 10

        analysis["fear_greed_score"] = max(0, min(100, fear_score))
        analysis["market_sentiment"] = (
            "extreme_fear" if fear_score > 80 else
            "fear" if fear_score > 60 else
            "neutral" if fear_score > 40 else
            "greed" if fear_score > 20 else
            "extreme_greed"
        )

        return analysis

    def get_vix_percentile(self, vix_value: float, lookback_days: int = 252) -> DataResult:
        """Calculate VIX percentile rank"""
        try:
            from investigator.infrastructure.database.db import get_db_manager
            from sqlalchemy import text
            from datetime import timedelta

            engine = get_db_manager().engine
            start_date = date.today() - timedelta(days=lookback_days)

            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT v.value
                        FROM macro_indicator_values v
                        JOIN macro_indicators i ON v.indicator_id = i.id
                        WHERE i.series_id = 'VIXCLS'
                        AND v.date >= :start_date
                        ORDER BY v.value
                    """),
                    {"start_date": start_date}
                )
                values = [float(r[0]) for r in result]

            if not values:
                return DataResult(success=False, error="No VIX history", source=self.name)

            # Calculate percentile
            below_count = sum(1 for v in values if v < vix_value)
            percentile = (below_count / len(values)) * 100

            return DataResult(
                success=True,
                data={
                    "vix_value": vix_value,
                    "percentile": round(percentile, 1),
                    "lookback_days": lookback_days,
                    "sample_size": len(values),
                    "min": min(values),
                    "max": max(values),
                    "median": sorted(values)[len(values) // 2],
                },
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)
