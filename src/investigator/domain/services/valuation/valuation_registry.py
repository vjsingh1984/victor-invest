"""
Valuation Registry - Plugin-based sector valuation handler registration.

Implements Registry pattern (SOLID: Open/Closed Principle) for sector-specific
valuation handlers. Allows new sectors to be added without modifying existing code.

Key Features:
1. Handler registration by sector/industry
2. Flexible pattern matching (substring, regex)
3. Priority-based selection when multiple handlers match
4. Default handler fallback

Usage:
    from investigator.domain.services.valuation.valuation_registry import (
        ValuationRegistry,
        get_valuation_registry,
    )

    # Get singleton registry
    registry = get_valuation_registry()

    # Register a new sector handler
    registry.register(
        sectors=["Financials", "Finance"],
        industries=["Insurance", "*insur*"],  # Wildcard pattern
        handler=insurance_valuation_handler,
        priority=10,
    )

    # Get handler for a company
    handler = registry.get_handler(sector="Financials", industry="Property-Casualty Insurers")

Author: InvestiGator Team
Date: 2025-01-05
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

logger = logging.getLogger(__name__)


class MatchType(Enum):
    """Type of pattern matching for industry/sector."""

    EXACT = "exact"  # Exact string match
    CONTAINS = "contains"  # Substring match (case-insensitive)
    REGEX = "regex"  # Regular expression match
    ANY = "any"  # Match any value (wildcard)


@dataclass
class HandlerRegistration:
    """Registration entry for a valuation handler."""

    handler_name: str
    handler: Callable
    sectors: Set[str]
    industries: Set[str]
    sector_patterns: List[Tuple[MatchType, Union[str, Pattern]]] = field(default_factory=list)
    industry_patterns: List[Tuple[MatchType, Union[str, Pattern]]] = field(default_factory=list)
    priority: int = 0  # Higher priority = checked first
    description: str = ""

    def matches(self, sector: Optional[str], industry: Optional[str]) -> bool:
        """
        Check if this handler matches the given sector/industry.

        Matching logic:
        1. Check exact sector match
        2. Check sector patterns (contains, regex)
        3. Check exact industry match
        4. Check industry patterns (contains, regex)

        Returns True if ANY sector criterion AND ANY industry criterion match.
        If no sectors defined, sector always matches.
        If no industries defined, industry always matches.
        """
        sector_match = self._matches_sector(sector)
        industry_match = self._matches_industry(industry)

        return sector_match and industry_match

    def _matches_sector(self, sector: Optional[str]) -> bool:
        """Check if sector matches any registered patterns."""
        # If no sectors registered, match all
        if not self.sectors and not self.sector_patterns:
            return True

        if sector is None:
            return False

        # Check exact match
        if sector in self.sectors:
            return True

        # Check patterns
        sector_lower = sector.lower()
        for match_type, pattern in self.sector_patterns:
            if self._pattern_matches(sector, sector_lower, match_type, pattern):
                return True

        return False

    def _matches_industry(self, industry: Optional[str]) -> bool:
        """Check if industry matches any registered patterns."""
        # If no industries registered, match all
        if not self.industries and not self.industry_patterns:
            return True

        if industry is None:
            return False

        # Check exact match
        if industry in self.industries:
            return True

        # Check patterns
        industry_lower = industry.lower()
        for match_type, pattern in self.industry_patterns:
            if self._pattern_matches(industry, industry_lower, match_type, pattern):
                return True

        return False

    @staticmethod
    def _pattern_matches(
        value: str,
        value_lower: str,
        match_type: MatchType,
        pattern: Union[str, Pattern],
    ) -> bool:
        """Check if value matches a pattern."""
        if match_type == MatchType.EXACT:
            return value == pattern

        elif match_type == MatchType.CONTAINS:
            return pattern.lower() in value_lower

        elif match_type == MatchType.REGEX:
            if isinstance(pattern, str):
                pattern = re.compile(pattern, re.IGNORECASE)
            return bool(pattern.search(value))

        elif match_type == MatchType.ANY:
            return True

        return False


class ValuationRegistry:
    """
    Registry for sector-specific valuation handlers.

    Implements Open/Closed Principle:
    - Open for extension: New handlers can be registered without modifying existing code
    - Closed for modification: Core registry logic doesn't change when handlers are added

    Design Patterns:
    - Registry: Central lookup for handlers by sector/industry
    - Strategy: Handlers implement different valuation strategies
    - Plugin: Handlers can be registered from external modules
    """

    def __init__(self):
        """Initialize empty registry."""
        self._handlers: List[HandlerRegistration] = []
        self._default_handler: Optional[Callable] = None
        self._initialized = False
        self.logger = logger

    def register(
        self,
        handler_name: str,
        handler: Callable,
        sectors: Optional[List[str]] = None,
        industries: Optional[List[str]] = None,
        priority: int = 0,
        description: str = "",
    ) -> "ValuationRegistry":
        """
        Register a valuation handler for specific sectors/industries.

        Args:
            handler_name: Unique identifier for the handler
            handler: Callable that performs valuation
            sectors: List of sectors (exact match or patterns with wildcards)
            industries: List of industries (exact match or patterns with wildcards)
            priority: Higher priority handlers are checked first (default: 0)
            description: Human-readable description of the handler

        Pattern syntax:
            - "Insurance" -> exact match
            - "*insur*" -> contains "insur" (case-insensitive)
            - "r:^Property.*" -> regex pattern

        Returns:
            Self for method chaining

        Example:
            >>> registry.register(
            ...     handler_name="insurance",
            ...     handler=value_insurance_company,
            ...     sectors=["Financials", "Finance"],
            ...     industries=["*insur*"],  # Matches any industry containing "insur"
            ...     priority=10,
            ...     description="Insurance company P/BV valuation"
            ... )
        """
        # Parse sectors into exact matches and patterns
        exact_sectors = set()
        sector_patterns = []
        for s in sectors or []:
            match_type, pattern = self._parse_pattern(s)
            if match_type == MatchType.EXACT:
                exact_sectors.add(pattern)
            else:
                sector_patterns.append((match_type, pattern))

        # Parse industries into exact matches and patterns
        exact_industries = set()
        industry_patterns = []
        for i in industries or []:
            match_type, pattern = self._parse_pattern(i)
            if match_type == MatchType.EXACT:
                exact_industries.add(pattern)
            else:
                industry_patterns.append((match_type, pattern))

        registration = HandlerRegistration(
            handler_name=handler_name,
            handler=handler,
            sectors=exact_sectors,
            industries=exact_industries,
            sector_patterns=sector_patterns,
            industry_patterns=industry_patterns,
            priority=priority,
            description=description,
        )

        self._handlers.append(registration)

        # Keep handlers sorted by priority (highest first)
        self._handlers.sort(key=lambda h: h.priority, reverse=True)

        self.logger.debug(
            f"Registered valuation handler: {handler_name} "
            f"(sectors={sectors}, industries={industries}, priority={priority})"
        )

        return self

    def register_default(self, handler: Callable) -> "ValuationRegistry":
        """
        Register the default handler for unmatched sectors/industries.

        Args:
            handler: Default valuation handler

        Returns:
            Self for method chaining
        """
        self._default_handler = handler
        self.logger.debug("Registered default valuation handler")
        return self

    def get_handler(
        self,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Callable]]:
        """
        Get the appropriate valuation handler for a sector/industry.

        Args:
            sector: Company sector
            industry: Company industry
            symbol: Stock symbol (for logging)

        Returns:
            Tuple of (handler_name, handler) or (None, default_handler)
        """
        for registration in self._handlers:
            if registration.matches(sector, industry):
                self.logger.debug(
                    f"{symbol or 'Unknown'} - Matched handler: {registration.handler_name} "
                    f"(sector={sector}, industry={industry})"
                )
                return (registration.handler_name, registration.handler)

        # No match, return default
        if self._default_handler:
            self.logger.debug(
                f"{symbol or 'Unknown'} - Using default handler " f"(sector={sector}, industry={industry})"
            )
            return (None, self._default_handler)

        return (None, None)

    def get_handler_info(self, handler_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered handler.

        Args:
            handler_name: Name of the handler

        Returns:
            Dict with handler info or None if not found
        """
        for reg in self._handlers:
            if reg.handler_name == handler_name:
                return {
                    "name": reg.handler_name,
                    "sectors": list(reg.sectors),
                    "industries": list(reg.industries),
                    "priority": reg.priority,
                    "description": reg.description,
                }
        return None

    def list_handlers(self) -> List[Dict[str, Any]]:
        """
        List all registered handlers.

        Returns:
            List of handler info dictionaries
        """
        return [
            {
                "name": reg.handler_name,
                "sectors": list(reg.sectors),
                "industries": list(reg.industries),
                "priority": reg.priority,
                "description": reg.description,
            }
            for reg in self._handlers
        ]

    def unregister(self, handler_name: str) -> bool:
        """
        Unregister a handler by name.

        Args:
            handler_name: Name of handler to remove

        Returns:
            True if handler was found and removed
        """
        for i, reg in enumerate(self._handlers):
            if reg.handler_name == handler_name:
                del self._handlers[i]
                self.logger.debug(f"Unregistered valuation handler: {handler_name}")
                return True
        return False

    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._default_handler = None
        self._initialized = False
        self.logger.debug("Cleared all valuation handlers")

    @staticmethod
    def _parse_pattern(pattern: str) -> Tuple[MatchType, Union[str, Pattern]]:
        """
        Parse a pattern string into match type and pattern.

        Pattern syntax:
            - "Insurance" -> EXACT, "Insurance"
            - "*insur*" -> CONTAINS, "insur"
            - "r:^Property.*" -> REGEX, compiled pattern
            - "*" -> ANY, None

        Args:
            pattern: Pattern string

        Returns:
            Tuple of (MatchType, pattern value)
        """
        if pattern == "*":
            return (MatchType.ANY, "")

        if pattern.startswith("r:"):
            # Regex pattern
            regex = pattern[2:]
            return (MatchType.REGEX, re.compile(regex, re.IGNORECASE))

        if pattern.startswith("*") and pattern.endswith("*"):
            # Contains pattern
            return (MatchType.CONTAINS, pattern[1:-1])

        if pattern.startswith("*"):
            # Ends with pattern -> regex
            suffix = pattern[1:]
            return (MatchType.REGEX, re.compile(f".*{re.escape(suffix)}$", re.IGNORECASE))

        if pattern.endswith("*"):
            # Starts with pattern -> regex
            prefix = pattern[:-1]
            return (MatchType.REGEX, re.compile(f"^{re.escape(prefix)}.*", re.IGNORECASE))

        # Exact match
        return (MatchType.EXACT, pattern)


# Singleton instance
_registry_instance: Optional[ValuationRegistry] = None


def get_valuation_registry() -> ValuationRegistry:
    """
    Get the singleton ValuationRegistry instance.

    The registry is lazily initialized with default handlers on first access.

    Returns:
        ValuationRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ValuationRegistry()
        _initialize_default_handlers(_registry_instance)
    return _registry_instance


def _initialize_default_handlers(registry: ValuationRegistry) -> None:
    """
    Initialize default sector handlers.

    This sets up the standard handlers for known sectors without
    directly depending on the handler implementations (lazy imports).
    """
    # Note: Actual handler implementations are imported lazily
    # to avoid circular imports. This just registers the routing rules.

    # Insurance (highest priority for financials)
    registry.register(
        handler_name="insurance",
        handler=_lazy_insurance_handler,
        sectors=["Financials", "Finance", "Financial Services"],
        industries=["*insur*", "Insurance"],
        priority=20,
        description="Insurance company P/BV valuation with combined ratio analysis",
    )

    # Banks
    registry.register(
        handler_name="bank",
        handler=_lazy_bank_handler,
        sectors=["Financials", "Finance", "Financial Services"],
        industries=["*bank*", "Banks", "Commercial Banks", "Regional Banks"],
        priority=15,
        description="Bank ROE-based valuation with NIM and efficiency metrics",
    )

    # REITs
    registry.register(
        handler_name="reit",
        handler=_lazy_reit_handler,
        sectors=["Real Estate"],
        industries=["*REIT*", "*Real Estate Investment Trust*"],
        priority=15,
        description="REIT FFO-multiple valuation by property type",
    )

    # Biotech (pre-revenue)
    registry.register(
        handler_name="biotech",
        handler=_lazy_biotech_handler,
        sectors=["Healthcare"],
        industries=[
            "Biotechnology",
            "Biopharmaceuticals",
            "Biological Products",
            "*biotech*",
        ],
        priority=10,
        description="Biotech pipeline probability-weighted valuation",
    )

    # Defense contractors
    registry.register(
        handler_name="defense",
        handler=_lazy_defense_handler,
        sectors=["Industrials"],
        industries=[
            "Aerospace & Defense",
            "Defense",
            "*defense*",
            "*military*",
            "Government Services",
        ],
        priority=10,
        description="Defense contractor backlog-adjusted valuation",
    )

    # Semiconductors
    registry.register(
        handler_name="semiconductor",
        handler=_lazy_semiconductor_handler,
        sectors=["Technology", "Information Technology"],
        industries=[
            "Semiconductors",
            "*semiconductor*",
            "*chip*",
            "Semiconductor Equipment",
        ],
        priority=10,
        description="Semiconductor cycle-adjusted valuation",
    )

    registry._initialized = True
    logger.info(f"Initialized valuation registry with {len(registry._handlers)} handlers")


# Lazy handler wrappers to avoid circular imports
def _lazy_insurance_handler(**kwargs) -> Any:
    """Lazy wrapper for insurance valuation."""
    from investigator.domain.services.valuation.insurance_valuation import (
        value_insurance_company,
    )

    return value_insurance_company(**kwargs)


def _lazy_bank_handler(**kwargs) -> Any:
    """Lazy wrapper for bank valuation."""
    from investigator.domain.services.valuation.bank_valuation import value_bank

    return value_bank(**kwargs)


def _lazy_reit_handler(**kwargs) -> Any:
    """Lazy wrapper for REIT valuation."""
    from investigator.domain.services.valuation.reit_valuation import value_reit

    return value_reit(**kwargs)


def _lazy_biotech_handler(**kwargs) -> Any:
    """Lazy wrapper for biotech valuation."""
    from investigator.domain.services.valuation.biotech_valuation import value_biotech

    return value_biotech(**kwargs)


def _lazy_defense_handler(**kwargs) -> Any:
    """Lazy wrapper for defense valuation."""
    from investigator.domain.services.valuation.defense_valuation import (
        value_defense_contractor,
    )

    return value_defense_contractor(**kwargs)


def _lazy_semiconductor_handler(**kwargs) -> Any:
    """Lazy wrapper for semiconductor valuation."""
    from investigator.domain.services.valuation.semiconductor_valuation import (
        value_semiconductor,
    )

    return value_semiconductor(**kwargs)
