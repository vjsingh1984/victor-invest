"""
SEC Data Infrastructure

SEC EDGAR filing processors and extractors.

Migration Date: 2025-11-14 (SEC API infrastructure)

NOTE: Some imports are done lazily to avoid circular import issues.
The circular dependency chain is:
  sec/__init__.py -> quarterly_processor -> data.models -> utils.sec_quarterly_processor
  -> utils.submission_processor -> investigator.application -> domain.agents.sec
  -> investigator.infrastructure.sec (circular!)

Direct imports from module files (e.g., `from investigator.infrastructure.sec.sec_api import SECApiClient`)
are preferred over package imports when possible.
"""


def __getattr__(name):
    """Lazy import to avoid circular dependencies.

    The following imports trigger circular dependency chains and must be lazy:
    - SECApiClient: sec_api -> utils.submission_processor -> application -> domain.agents.sec -> sec (circular)
    - SECQuarterlyProcessor: quarterly_processor -> data.models -> utils -> application -> domain.agents.sec -> sec
    - SECDataProcessor: data_processor -> metric_extraction -> ... -> sec
    - SECCompanyFactsExtractor: companyfacts_extractor -> domain services -> ... -> sec
    """
    if name == "SECQuarterlyProcessor":
        from investigator.infrastructure.sec.quarterly_processor import SECQuarterlyProcessor
        return SECQuarterlyProcessor
    elif name == "SECDataProcessor":
        from investigator.infrastructure.sec.data_processor import SECDataProcessor
        return SECDataProcessor
    elif name == "SECCompanyFactsExtractor":
        from investigator.infrastructure.sec.companyfacts_extractor import SECCompanyFactsExtractor
        return SECCompanyFactsExtractor
    elif name == "SECApiClient":
        from investigator.infrastructure.sec.sec_api import SECApiClient
        return SECApiClient
    elif name == "SECAPIClient":  # Backwards compatibility alias
        from investigator.infrastructure.sec.sec_api import SECApiClient
        return SECApiClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# These imports are safe and don't trigger circular dependencies
from investigator.infrastructure.sec.canonical_mapper import (
    CanonicalKeyMapper,
    get_canonical_mapper,
)
from investigator.infrastructure.sec.data_strategy import SECDataStrategy
from investigator.infrastructure.sec.sec_frame_api import SECFrameAPI, get_frame_api
from investigator.infrastructure.sec.xbrl_parser import XBRLParser

__all__ = [
    "CanonicalKeyMapper",
    "get_canonical_mapper",
    "SECApiClient",
    "SECCompanyFactsExtractor",
    "SECDataProcessor",
    "SECDataStrategy",
    "SECFrameAPI",
    "SECQuarterlyProcessor",
    "XBRLParser",
    "get_frame_api",
]
