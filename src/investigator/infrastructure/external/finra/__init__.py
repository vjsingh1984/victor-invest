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

"""FINRA Data Infrastructure Module.

This module provides access to FINRA market data including:
- Short interest data
- Short volume data
- Trading activity reports

Data Source: https://api.finra.org/data/
"""

from investigator.infrastructure.external.finra.short_interest import (
    ShortInterestFetcher,
    ShortInterestData,
    ShortVolumeData,
    get_short_interest_fetcher,
)

__all__ = [
    "ShortInterestFetcher",
    "ShortInterestData",
    "ShortVolumeData",
    "get_short_interest_fetcher",
]
