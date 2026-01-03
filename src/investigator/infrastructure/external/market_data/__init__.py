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

"""Additional Market Data Sources.

Provides access to:
- CBOE: VIX term structure, options data
- ISM: PMI manufacturing and services (via FRED proxy)
- BLS: Employment data, CPI details
- Census: Retail sales, housing starts
"""

from .cboe_data import (
    CBOEClient,
    VIXTermStructure,
    get_cboe_client,
)

__all__ = [
    "CBOEClient",
    "VIXTermStructure",
    "get_cboe_client",
]
