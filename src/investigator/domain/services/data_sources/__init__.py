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

"""Unified Data Sources Module.

Provides consistent access to all data sources for:
- CLI analysis tools
- RL backtesting
- Batch processing
- Scheduled collectors

All data sources follow SOLID principles:
- Single Responsibility: Each source handles one type of data
- Open/Closed: Easy to add new sources without modifying existing
- Liskov Substitution: All sources implement common interface
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: High-level modules depend on abstractions
"""

from investigator.domain.services.data_sources.facade import (
    DataSourceFacade,
    get_data_source_facade,
)
from investigator.domain.services.data_sources.interfaces import (
    DataSourceInterface,
    DataSourceResult,
)

__all__ = [
    "DataSourceFacade",
    "get_data_source_facade",
    "DataSourceInterface",
    "DataSourceResult",
]
