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

"""NY Fed Markets Data Infrastructure.

Provides access to New York Federal Reserve economic indicators including:
- Recession probability
- Global Supply Chain Pressure Index (GSCPI)
- Term premia estimates

Example:
    from investigator.infrastructure.external.nyfed import (
        NYFedDataClient,
        get_nyfed_client,
    )

    client = get_nyfed_client()
    recession_prob = await client.get_recession_probability()
"""

from investigator.infrastructure.external.nyfed.markets_data import (
    GSCPIData,
    NYFedDataClient,
    RecessionProbability,
    get_nyfed_client,
)

__all__ = [
    "NYFedDataClient",
    "RecessionProbability",
    "GSCPIData",
    "get_nyfed_client",
]
