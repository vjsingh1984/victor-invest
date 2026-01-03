"""
Concrete Data Source Implementations

Each source follows SOLID principles:
- Single Responsibility: One data type per source
- Open/Closed: Extend base classes, don't modify
- Liskov Substitution: All sources are interchangeable
- Interface Segregation: Implement only needed interfaces
- Dependency Inversion: Depend on abstractions
"""

from .fred import FredMacroSource
from .treasury import TreasuryYieldSource
from .cboe import CBOEVolatilitySource
from .fed_districts import (
    AtlantaFedSource,
    ChicagoFedSource,
    ClevelandFedSource,
    DallasFedSource,
    KansasCityFedSource,
    NewYorkFedSource,
    PhiladelphiaFedSource,
    RichmondFedSource,
)
from .sec import (
    InsiderTransactionSource,
    InstitutionalHoldingsSource,
    SECQuarterlySource,
)
from .market import (
    PriceHistorySource,
    TechnicalIndicatorSource,
    ShortInterestSource,
)

__all__ = [
    # FRED
    "FredMacroSource",
    # Treasury
    "TreasuryYieldSource",
    # CBOE
    "CBOEVolatilitySource",
    # Fed Districts
    "AtlantaFedSource",
    "ChicagoFedSource",
    "ClevelandFedSource",
    "DallasFedSource",
    "KansasCityFedSource",
    "NewYorkFedSource",
    "PhiladelphiaFedSource",
    "RichmondFedSource",
    # SEC
    "InsiderTransactionSource",
    "InstitutionalHoldingsSource",
    "SECQuarterlySource",
    # Market
    "PriceHistorySource",
    "TechnicalIndicatorSource",
    "ShortInterestSource",
]
