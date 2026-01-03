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

"""Federal Reserve District Banks Data Collectors.

This module provides access to economic data from all 12 Federal Reserve
district banks. Each district publishes unique regional economic indicators
that provide valuable signals for investment research.

Federal Reserve Districts:
    1. Boston Fed - New England economic indicators
    2. New York Fed - Recession probability, GSCPI, Empire State Manufacturing
    3. Philadelphia Fed - Manufacturing survey, Leading Index, Coincident Index
    4. Cleveland Fed - Inflation expectations, Yield Curve Model
    5. Richmond Fed - Fifth District surveys
    6. Atlanta Fed - GDPNow, Business surveys
    7. Chicago Fed - CFNAI, Midwest economy
    8. St. Louis Fed - FRED (covered separately)
    9. Minneapolis Fed - Ninth District data
    10. Kansas City Fed - Manufacturing survey
    11. Dallas Fed - Texas business surveys
    12. San Francisco Fed - Western economy, research

Key Investment Signals:
    - GDPNow (Atlanta): Real-time GDP tracking
    - CFNAI (Chicago): Recession probability from national activity
    - Philly Fed Leading Index: 6-month forward outlook
    - Regional surveys: Early manufacturing/services signals
    - Inflation expectations (Cleveland): Forward inflation pricing
"""

from .atlanta_fed import AtlantaFedClient, GDPNowData, get_atlanta_fed_client
from .chicago_fed import CFNAIData, ChicagoFedClient, get_chicago_fed_client
from .cleveland_fed import ClevelandFedClient, InflationExpectations, get_cleveland_fed_client
from .dallas_fed import DallasFedClient, TexasManufacturing, get_dallas_fed_client
from .kansas_city_fed import KansasCityFedClient, KCManufacturing, get_kc_fed_client
from .philadelphia_fed import (
    LeadingIndex,
    ManufacturingSurvey,
    PhiladelphiaFedClient,
    get_philly_fed_client,
)
from .richmond_fed import FifthDistrictSurvey, RichmondFedClient, get_richmond_fed_client

__all__ = [
    # Atlanta Fed
    "AtlantaFedClient",
    "GDPNowData",
    "get_atlanta_fed_client",
    # Chicago Fed
    "ChicagoFedClient",
    "CFNAIData",
    "get_chicago_fed_client",
    # Cleveland Fed
    "ClevelandFedClient",
    "InflationExpectations",
    "get_cleveland_fed_client",
    # Dallas Fed
    "DallasFedClient",
    "TexasManufacturing",
    "get_dallas_fed_client",
    # Kansas City Fed
    "KansasCityFedClient",
    "KCManufacturing",
    "get_kc_fed_client",
    # Philadelphia Fed
    "PhiladelphiaFedClient",
    "ManufacturingSurvey",
    "LeadingIndex",
    "get_philly_fed_client",
    # Richmond Fed
    "RichmondFedClient",
    "FifthDistrictSurvey",
    "get_richmond_fed_client",
]
