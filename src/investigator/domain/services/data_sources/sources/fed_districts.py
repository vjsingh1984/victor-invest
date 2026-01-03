"""
Federal Reserve District Data Sources

Provides data from all 12 Federal Reserve districts with standardized interfaces.
Each district source follows the same pattern for consistency.
"""

from datetime import date
from typing import Any, Dict, List, Optional
import logging

from ..base import (
    MacroDataSource, DataResult, SourceMetadata,
    DataCategory, DataFrequency, DataQuality
)
from ..registry import register_source

logger = logging.getLogger(__name__)


class FedDistrictSource(MacroDataSource):
    """
    Base class for Federal Reserve district data sources.

    Provides common functionality for fetching district-specific indicators.
    """

    DISTRICT_NAME: str = ""
    INDICATORS: List[str] = []

    def __init__(self):
        super().__init__(f"{self.DISTRICT_NAME.lower().replace(' ', '_')}_fed", DataFrequency.DAILY)
        # District names in DB include "_fed" suffix (e.g., "chicago_fed")
        self.district = f"{self.DISTRICT_NAME.lower().replace(' ', '_')}_fed"

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name=self.name,
            category=DataCategory.MACRO,
            frequency=self.frequency,
            description=f"{self.DISTRICT_NAME} Federal Reserve indicators",
            provider=f"Federal Reserve Bank of {self.DISTRICT_NAME}",
            is_free=True,
            requires_api_key=False,
            lookback_days=365 * 10,
            symbols_supported=False,
        )

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Fetch district-specific indicators"""
        try:
            from investigator.infrastructure.database.db import get_db_manager
            from sqlalchemy import text

            engine = get_db_manager().engine
            target_date = as_of_date or date.today()

            data = {}
            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT indicator_name,
                               (indicator_data->>'value')::float as value,
                               observation_date,
                               indicator_data->>'fred_series' as fred_series
                        FROM regional_fed_indicators
                        WHERE district = :district
                        AND observation_date <= :target_date
                        ORDER BY indicator_name, observation_date DESC
                    """),
                    {"district": self.district, "target_date": target_date}
                )

                seen = set()
                for row in result:
                    indicator = row[0]
                    if indicator not in seen:
                        seen.add(indicator)
                        data[indicator] = {
                            "value": float(row[1]) if row[1] else None,
                            "date": row[2].isoformat() if row[2] else None,
                            "fred_series": row[3],
                        }

            if not data:
                return DataResult(
                    success=False,
                    error=f"No data for {self.DISTRICT_NAME} Fed",
                    source=self.name,
                )

            return DataResult(
                success=True,
                data={"district": self.DISTRICT_NAME, "indicators": data},
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            logger.error(f"{self.DISTRICT_NAME} Fed fetch error: {e}")
            return DataResult(success=False, error=str(e), source=self.name)


@register_source("atlanta_fed", DataCategory.MACRO)
class AtlantaFedSource(FedDistrictSource):
    """Atlanta Fed - GDPNow, Wage Growth Tracker"""
    DISTRICT_NAME = "Atlanta"
    INDICATORS = ["gdpnow", "wage_growth_tracker", "business_inflation_expectations"]


@register_source("chicago_fed", DataCategory.MACRO)
class ChicagoFedSource(FedDistrictSource):
    """Chicago Fed - CFNAI, NFCI"""
    DISTRICT_NAME = "Chicago"
    INDICATORS = ["cfnai", "nfci", "anfci"]


@register_source("cleveland_fed", DataCategory.MACRO)
class ClevelandFedSource(FedDistrictSource):
    """Cleveland Fed - Inflation Expectations, Yield Curve Model"""
    DISTRICT_NAME = "Cleveland"
    INDICATORS = ["inflation_expectations", "yield_curve_model", "median_cpi", "trimmed_mean_cpi"]


@register_source("dallas_fed", DataCategory.MACRO)
class DallasFedSource(FedDistrictSource):
    """Dallas Fed - Texas Manufacturing, Trimmed Mean PCE"""
    DISTRICT_NAME = "Dallas"
    INDICATORS = ["texas_manufacturing", "texas_services", "trimmed_mean_pce"]


@register_source("kansas_city_fed", DataCategory.MACRO)
class KansasCityFedSource(FedDistrictSource):
    """Kansas City Fed - Manufacturing Survey, KCFSI"""
    DISTRICT_NAME = "Kansas City"
    INDICATORS = ["manufacturing_survey", "kcfsi", "lmci"]

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        result = super()._fetch_impl(symbol, as_of_date)
        if result.success and result.data:
            # Add KCFSI interpretation
            indicators = result.data.get("indicators", {})
            if "kcfsi" in indicators:
                kcfsi = indicators["kcfsi"]["value"]
                if kcfsi is not None:
                    if kcfsi > 1.0:
                        result.data["stress_level"] = "high"
                    elif kcfsi > 0.5:
                        result.data["stress_level"] = "elevated"
                    elif kcfsi > 0:
                        result.data["stress_level"] = "above_average"
                    elif kcfsi > -0.5:
                        result.data["stress_level"] = "normal"
                    else:
                        result.data["stress_level"] = "low"
        return result


@register_source("new_york_fed", DataCategory.MACRO)
class NewYorkFedSource(FedDistrictSource):
    """New York Fed - Recession Probability, Empire State Mfg"""
    DISTRICT_NAME = "New York"
    INDICATORS = ["recession_probability", "gscpi", "empire_state_manufacturing"]

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        result = super()._fetch_impl(symbol, as_of_date)
        if result.success and result.data:
            indicators = result.data.get("indicators", {})
            # Add recession risk interpretation
            if "recession_probability" in indicators:
                prob = indicators["recession_probability"]["value"]
                if prob is not None:
                    if prob > 50:
                        result.data["recession_risk"] = "high"
                    elif prob > 30:
                        result.data["recession_risk"] = "elevated"
                    elif prob > 15:
                        result.data["recession_risk"] = "moderate"
                    else:
                        result.data["recession_risk"] = "low"
        return result


@register_source("philadelphia_fed", DataCategory.MACRO)
class PhiladelphiaFedSource(FedDistrictSource):
    """Philadelphia Fed - Manufacturing Survey, ADS Index"""
    DISTRICT_NAME = "Philadelphia"
    INDICATORS = ["manufacturing_survey", "ads_index", "leading_index", "coincident_index"]


@register_source("richmond_fed", DataCategory.MACRO)
class RichmondFedSource(FedDistrictSource):
    """Richmond Fed - Manufacturing Survey, Services Survey"""
    DISTRICT_NAME = "Richmond"
    INDICATORS = ["manufacturing_survey", "services_survey"]


# =============================================================================
# Composite Fed Source (All Districts)
# =============================================================================

@register_source("all_fed_districts", DataCategory.MACRO)
class AllFedDistrictsSource(MacroDataSource):
    """
    Aggregated view of all Federal Reserve district data.

    Provides:
    - Summary statistics across districts
    - Regional economic heat map
    - Consensus indicators
    """

    def __init__(self):
        super().__init__("all_fed_districts", DataFrequency.DAILY)
        self._district_sources = [
            AtlantaFedSource(),
            ChicagoFedSource(),
            ClevelandFedSource(),
            DallasFedSource(),
            KansasCityFedSource(),
            NewYorkFedSource(),
            PhiladelphiaFedSource(),
            RichmondFedSource(),
        ]

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="all_fed_districts",
            category=DataCategory.MACRO,
            frequency=DataFrequency.DAILY,
            description="Aggregated Federal Reserve district data",
            provider="Federal Reserve System",
            is_free=True,
            requires_api_key=False,
            lookback_days=365 * 10,
            symbols_supported=False,
        )

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Fetch and aggregate data from all districts"""
        by_district = {}
        summary = {}
        errors = []

        for source in self._district_sources:
            result = source.fetch(symbol, as_of_date)
            if result.success:
                district = result.data.get("district", source.DISTRICT_NAME)
                by_district[district.lower()] = result.data.get("indicators", {})

                # Extract key summary indicators
                indicators = result.data.get("indicators", {})
                for key in ["gdpnow", "cfnai", "nfci", "kcfsi", "recession_probability",
                            "inflation_expectations", "empire_state_manufacturing"]:
                    if key in indicators and indicators[key].get("value") is not None:
                        summary[key] = indicators[key]["value"]
            else:
                errors.append(f"{source.DISTRICT_NAME}: {result.error}")

        if not by_district:
            return DataResult(
                success=False,
                error=f"No district data: {'; '.join(errors)}",
                source=self.name,
            )

        return DataResult(
            success=True,
            data={
                "by_district": by_district,
                "summary": summary,
                "districts_available": list(by_district.keys()),
                "errors": errors if errors else None,
            },
            source=self.name,
            quality=DataQuality.HIGH if not errors else DataQuality.MEDIUM,
        )
