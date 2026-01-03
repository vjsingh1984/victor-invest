"""
Canonical Key Mapper - Sector-Aware XBRL Tag Resolution with Derivation Support

Maps universal financial concepts (canonical keys) to XBRL tags with:
- Sector/industry-specific fallback chains
- Dual support for JSON API and bulk table extraction
- Automatic tag resolution with priority ordering
- Derived/calculated metric support with formula evaluation
- Automatic dependency resolution for derived metrics
- Coverage reporting and gap detection

Author: InvestiGator Team
Date: 2025-11-03
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class CanonicalKeyMapper:
    """
    Maps canonical keys to XBRL tags with sector-aware fallback chains

    Example Usage:
        mapper = CanonicalKeyMapper()

        # Get tags for revenue (Technology sector)
        revenue_tags = mapper.get_tags('total_revenue', sector='Technology')
        # Returns: ['RevenueFromContractWithCustomer...', 'Revenues', 'SalesRevenueNet']

        # Extract value from JSON API data
        value = mapper.extract_from_json(
            canonical_key='total_revenue',
            json_data=companyfacts_json,
            sector='Technology'
        )
    """

    def __init__(self, mappings_path: Optional[str] = None, sector_mappings_path: Optional[str] = None):
        """
        Initialize mapper with tag mappings

        Args:
            mappings_path: Path to canonical_key_mappings.json
                          If None, uses default built-in mappings
            sector_mappings_path: Path to sector_specific_mappings.json
                                 If None, loads from default resources location
        """
        self.mappings = self._load_mappings(mappings_path)
        self.sector_mappings = self._load_sector_mappings(sector_mappings_path)
        self.stats = {
            'extractions': 0,
            'fallbacks_used': 0,
            'failures': 0
        }

    def _load_mappings(self, mappings_path: Optional[str]) -> Dict:
        """Load canonical key mappings from file or use defaults"""

        if mappings_path and Path(mappings_path).exists():
            with open(mappings_path, 'r') as f:
                return json.load(f)

        # Try to load comprehensive mappings with derivations from resources
        # Path resolution: src/investigator/infrastructure/sec/ → project root (5 levels up)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        comprehensive_with_derivations_path = project_root / 'resources' / 'xbrl_mappings' / 'comprehensive_canonical_mappings_with_derivations.json'
        if comprehensive_with_derivations_path.exists():
            logger.info(f"Loading comprehensive canonical mappings with derivations from {comprehensive_with_derivations_path}")
            with open(comprehensive_with_derivations_path, 'r') as f:
                return json.load(f)

        # Try to load comprehensive mappings (without derivations) from resources
        comprehensive_path = project_root / 'resources' / 'xbrl_mappings' / 'comprehensive_canonical_mappings.json'
        if comprehensive_path.exists():
            logger.info(f"Loading comprehensive canonical mappings from {comprehensive_path}")
            with open(comprehensive_path, 'r') as f:
                return json.load(f)

        # CRITICAL: Comprehensive mappings are required for production use
        error_msg = (
            f"CRITICAL: Comprehensive canonical mappings not found!\n"
            f"Expected location: {comprehensive_with_derivations_path}\n"
            f"Alternative location: {comprehensive_path}\n"
            f"The system requires comprehensive XBRL mappings (69 keys) for proper operation.\n"
            f"Falling back to minimal default mappings (10 keys) would severely degrade functionality.\n"
            f"Please ensure resources/xbrl_mappings/comprehensive_canonical_mappings_with_derivations.json exists."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    def _load_sector_mappings(self, sector_mappings_path: Optional[str]) -> Dict:
        """
        Load sector-specific XBRL mappings with industry/sector-specific tags and exclusions

        Args:
            sector_mappings_path: Path to sector_specific_mappings.json
                                 If None, loads from default resources location

        Returns:
            Dict with sector/industry-specific mappings, or empty dict if not found
        """
        if sector_mappings_path and Path(sector_mappings_path).exists():
            with open(sector_mappings_path, 'r') as f:
                return json.load(f)

        # Try to load from resources directory
        # Path resolution: src/investigator/infrastructure/sec/ → project root (5 levels up)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        default_path = project_root / 'resources' / 'xbrl_mappings' / 'sector_specific_mappings.json'
        if default_path.exists():
            logger.info(f"Loading sector-specific mappings from {default_path}")
            with open(default_path, 'r') as f:
                return json.load(f)

        logger.warning("Sector-specific mappings not found - sector/industry exclusions won't be applied")
        return {}

    def _get_default_mappings(self) -> Dict:
        """
        Get default canonical key → tag mappings

        NOTE: This is a starting point. Run scripts/analyze_sp100_xbrl_tags.py
        to generate comprehensive sector-specific mappings.
        """
        return {
            'total_revenue': {
                'description': 'Total Revenue/Sales',
                'global_fallback': [
                    'RevenueFromContractWithCustomerExcludingAssessedTax',
                    'RevenueFromContractWithCustomerIncludingAssessedTax',
                    'Revenues',
                    'SalesRevenueNet',
                ],
                'sector_specific': {
                    'Utilities': [
                        'RegulatedOperatingRevenue',
                        'ElectricUtilityRevenue',
                        'RevenuesExcludingInterestAndDividends',
                        'Revenues',
                    ],
                    'Financials': [
                        'InterestAndDividendIncomeOperating',
                        'InterestIncomeOperating',
                        'Revenues',
                    ],
                    'Real Estate': [
                        'RealEstateRevenueNet',
                        'Revenues',
                    ],
                }
            },

            'net_income': {
                'description': 'Net Income',
                'global_fallback': [
                    'NetIncomeLoss',
                    'NetIncomeLossAvailableToCommonStockholdersBasic',
                    'ProfitLoss',
                ],
                'sector_specific': {}
            },

            'operating_cash_flow': {
                'description': 'Cash from Operations',
                'global_fallback': [
                    'NetCashProvidedByUsedInOperatingActivities',
                    'NetCashFlowOperatingActivities',
                ],
                'sector_specific': {}
            },

            'capital_expenditures': {
                'description': 'Capital Expenditures',
                'global_fallback': [
                    'PaymentsToAcquirePropertyPlantAndEquipment',
                    'CapitalExpendituresIncurredButNotYetPaid',
                    'PaymentsForCapitalImprovements',
                ],
                'sector_specific': {
                    'Utilities': [
                        'PaymentsToAcquirePropertyPlantAndEquipment',
                        'CapitalExpendituresIncurredButNotYetPaid',
                    ],
                }
            },

            'dividends_paid': {
                'description': 'Dividends Paid',
                'global_fallback': [
                    'PaymentsOfDividends',
                    'PaymentsOfDividendsCommonStock',
                    'PaymentsOfOrdinaryDividends',
                    'DividendsCashOutflow',
                ],
                'sector_specific': {
                    'Real Estate': [
                        'PaymentsOfDistributionsToAffiliates',
                        'PaymentsOfDividends',
                    ],
                }
            },

            'total_assets': {
                'description': 'Total Assets',
                'global_fallback': [
                    'Assets',
                    'AssetsCurrent',  # Fallback if only current assets available
                ],
                'sector_specific': {}
            },

            'total_liabilities': {
                'description': 'Total Liabilities',
                'global_fallback': [
                    'Liabilities',
                    'LiabilitiesCurrent',  # Fallback
                    'LiabilitiesAndStockholdersEquity',
                    'LiabilitiesNoncurrent',
                ],
                'sector_specific': {},
                'derived': {
                    'enabled': True,
                    'formula': 'total_assets - stockholders_equity',
                    'required_fields': ['total_assets', 'stockholders_equity'],
                    'description': 'Derive liabilities from Assets - Equity when direct tag missing'
                }
            },

            'stockholders_equity': {
                'description': 'Shareholders Equity',
                'global_fallback': [
                    'StockholdersEquity',
                    'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                    'StockholdersEquityDeficit',
                ],
                'sector_specific': {},
                'derived': {
                    'enabled': True,
                    'formula': 'total_assets - total_liabilities',
                    'required_fields': ['total_assets', 'total_liabilities'],
                    'description': 'Fallback accounting identity when direct tag missing'
                }
            },

            # Add more canonical keys as needed...
        }

    def get_tags(
        self,
        canonical_key: str,
        sector: Optional[str] = None,
        industry: Optional[str] = None
    ) -> List[str]:
        """
        Get priority-ordered list of XBRL tags for a canonical key

        Priority order: industry_specific > sector_specific > global_fallback

        Args:
            canonical_key: Financial concept (e.g., 'total_revenue', 'revenues')
            sector: Company sector (e.g., 'Financials', 'Technology')
            industry: Company industry (e.g., 'Banks', 'Insurance')

        Returns:
            List of XBRL tags in priority order (try first → last)
        """
        tags = []

        # PRIORITY 1: Industry-specific tags from sector_mappings (highest priority)
        if self.sector_mappings and sector and industry:
            sector_config = self.sector_mappings.get(sector, {})
            tag_mappings = sector_config.get('tag_mappings', {})

            if canonical_key in tag_mappings:
                mapping = tag_mappings[canonical_key]

                # Check if this mapping has industry-specific tags
                if isinstance(mapping, dict) and industry in mapping:
                    industry_tags = mapping[industry]
                    if isinstance(industry_tags, list):
                        tags.extend(industry_tags)
                        logger.debug(
                            f"Using industry-specific tags for {canonical_key} "
                            f"(sector={sector}, industry={industry}): {industry_tags}"
                        )
                # Fall back to sector's _default if no industry match
                elif isinstance(mapping, dict) and '_default' in mapping:
                    default_tags = mapping['_default']
                    if isinstance(default_tags, list):
                        tags.extend(default_tags)

        # PRIORITY 2: Sector-specific tags from sector_mappings
        if self.sector_mappings and sector and not tags:
            sector_config = self.sector_mappings.get(sector, {})
            tag_mappings = sector_config.get('tag_mappings', {})

            if canonical_key in tag_mappings:
                mapping = tag_mappings[canonical_key]

                # If mapping is a simple list (not industry-specific)
                if isinstance(mapping, list):
                    tags.extend(mapping)
                # If mapping has _default but no industry was provided
                elif isinstance(mapping, dict) and '_default' in mapping:
                    default_tags = mapping['_default']
                    if isinstance(default_tags, list):
                        tags.extend(default_tags)

        # PRIORITY 3: Global fallback from sector_mappings
        if self.sector_mappings and not tags:
            global_config = self.sector_mappings.get('_global', {})
            tag_mappings = global_config.get('tag_mappings', {})

            if canonical_key in tag_mappings:
                global_tags = tag_mappings[canonical_key]
                if isinstance(global_tags, list):
                    tags.extend(global_tags)

        # PRIORITY 4: Standard mappings (comprehensive_canonical_mappings.json)
        if canonical_key in self.mappings:
            mapping = self.mappings[canonical_key]

            # Add sector-specific tags from standard mappings
            if sector and sector in mapping.get('sector_specific', {}):
                tags.extend(mapping['sector_specific'][sector])

            # Add global fallback from standard mappings
            tags.extend(mapping.get('global_fallback', []))

        # If no tags found anywhere
        if not tags:
            logger.warning(
                f"No tags found for canonical key '{canonical_key}' "
                f"(sector={sector}, industry={industry})"
            )
            return []

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        return unique_tags

    def extract_from_json(
        self,
        canonical_key: str,
        json_data: Dict,
        sector: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract value for canonical key from SEC CompanyFacts JSON

        Args:
            canonical_key: Financial concept to extract
            json_data: SEC CompanyFacts JSON response
            sector: Company sector for sector-specific tag resolution
            fiscal_year: Filter to specific fiscal year
            fiscal_period: Filter to specific fiscal period (Q1, Q2, Q3, Q4, FY)

        Returns:
            Tuple of (value, tag_used) or (None, None) if not found
        """
        tags = self.get_tags(canonical_key, sector=sector)

        if not tags:
            logger.warning(f"No tags mapped for canonical key: {canonical_key}")
            return (None, None)

        us_gaap = json_data.get('facts', {}).get('us-gaap', {})

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "🔍 Fallback chain for %s (sector=%s): %s",
                canonical_key,
                sector or "global",
                tags,
            )

        for i, tag in enumerate(tags):
            if tag not in us_gaap:
                logger.debug("↪️ Skipping tag '%s' for %s (not present in us-gaap)", tag, canonical_key)
                continue

            # Extract latest value for this tag
            units = us_gaap[tag].get('units', {})

            # Try USD first, then other currencies
            for currency in ['USD', 'EUR', 'GBP']:
                if currency not in units:
                    continue

                entries = units[currency]

                # Filter by fiscal period if specified
                if fiscal_year or fiscal_period:
                    entries = [
                        e for e in entries
                        if (not fiscal_year or e.get('fy') == fiscal_year)
                        and (not fiscal_period or e.get('fp') == fiscal_period)
                    ]

                if not entries:
                    continue

                # Sort by filed date (most recent first)
                entries.sort(key=lambda x: x.get('filed', ''), reverse=True)

                value = entries[0].get('val')

                if value is not None:
                    self.stats['extractions'] += 1
                    logger.debug(
                        "✅ %s extracted using tag '%s' (attempt %d of %d)",
                        canonical_key,
                        tag,
                        i + 1,
                        len(tags),
                    )
                    if i > 0:
                        self.stats['fallbacks_used'] += 1
                        logger.debug(
                            f"🎯 Fallback SUCCESS: {canonical_key} → {tag} "
                            f"(tried {i} tags before success)"
                        )

                    return (float(value), tag)

        # No value found in any fallback
        self.stats['failures'] += 1
        logger.warning(
            f"⚠️  Failed to extract {canonical_key} for sector {sector}. "
            f"Tried tags: {tags}"
        )
        return (None, None)

    def extract_from_bulk_table(
        self,
        canonical_key: str,
        tag_values: Dict[str, float],
        sector: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract value for canonical key from bulk table tag values

        Args:
            canonical_key: Financial concept to extract
            tag_values: Dict mapping XBRL tags to values (from sec_num_data)
            sector: Company sector for sector-specific tag resolution

        Returns:
            Tuple of (value, tag_used) or (None, None) if not found
        """
        tags = self.get_tags(canonical_key, sector=sector)

        if not tags:
            logger.warning(f"No tags mapped for canonical key: {canonical_key}")
            return (None, None)

        for i, tag in enumerate(tags):
            if tag in tag_values and tag_values[tag] is not None:
                value = tag_values[tag]

                self.stats['extractions'] += 1
                logger.debug(
                    "✅ %s extracted from bulk table using tag '%s' (attempt %d of %d)",
                    canonical_key,
                    tag,
                    i + 1,
                    len(tags),
                )
                if i > 0:
                    self.stats['fallbacks_used'] += 1
                    logger.debug(
                        f"🎯 Bulk table fallback SUCCESS: {canonical_key} → {tag} "
                        f"(tried {i} tags)"
                    )

                return (float(value), tag)

        # No value found
        self.stats['failures'] += 1
        logger.warning(
            f"⚠️  Bulk table: Failed to extract {canonical_key}. "
            f"Tried tags: {tags}, Available tags: {list(tag_values.keys())}"
        )
        return (None, None)

    def _parse_formula(self, formula: str, values_dict: Dict[str, float]) -> Optional[float]:
        """
        Parse and evaluate a formula string using provided values

        Args:
            formula: Formula string like "total_revenue - gross_profit" or "total_assets - stockholders_equity"
            values_dict: Dictionary of canonical_key -> value mappings

        Returns:
            Calculated result or None if calculation fails
        """
        try:
            # Simple formula parser for basic arithmetic
            # Supports: +, -, *, /

            # Replace canonical keys with their values
            formula_eval = formula
            for key, value in values_dict.items():
                if value is not None:
                    formula_eval = formula_eval.replace(key, str(value))

            # Check if all variables were replaced (no remaining canonical keys)
            if any(key in formula_eval for key in values_dict.keys() if key in self.mappings):
                # Still has unresolved canonical keys
                logger.debug(f"Cannot evaluate formula '{formula}' - missing required values")
                return None

            # Evaluate the formula
            result = eval(formula_eval, {"__builtins__": {}}, {})
            return float(result)

        except Exception as e:
            logger.debug(f"Formula evaluation failed for '{formula}': {e}")
            return None

    def calculate_derived_value(
        self,
        canonical_key: str,
        values_dict: Dict[str, float]
    ) -> Optional[float]:
        """
        Calculate derived value using formula from mapping

        Args:
            canonical_key: Financial concept to calculate (e.g., 'free_cash_flow')
            values_dict: Dictionary of available canonical_key -> value mappings

        Returns:
            Calculated value or None if derivation fails
        """
        if canonical_key not in self.mappings:
            return None

        mapping = self.mappings[canonical_key]
        derived_config = mapping.get('derived', {})

        if not derived_config.get('enabled', False):
            return None

        # Try primary formula
        formula = derived_config.get('formula')
        if formula:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "🧮 Attempting derivation for %s using formula \"%s\" with inputs: %s",
                    canonical_key,
                    formula,
                    {k: values_dict.get(k) for k in values_dict.keys()},
                )
            result = self._parse_formula(formula, values_dict)
            if result is not None:
                logger.debug(f"✓ Derived {canonical_key} using formula: {formula} = {result}")
                return result

        # Try alternative formula
        formula_alt = derived_config.get('formula_alt')
        if formula_alt:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "🧮 Attempting derivation for %s using alternate formula \"%s\" with inputs: %s",
                    canonical_key,
                    formula_alt,
                    {k: values_dict.get(k) for k in values_dict.keys()},
                )
            result = self._parse_formula(formula_alt, values_dict)
            if result is not None:
                logger.debug(f"✓ Derived {canonical_key} using alt formula: {formula_alt} = {result}")
                return result

        logger.debug(f"Cannot derive {canonical_key} - formulas failed or missing required fields")
        return None

    def extract_with_derivation(
        self,
        canonical_key: str,
        json_data: Optional[Dict] = None,
        tag_values: Optional[Dict[str, float]] = None,
        sector: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        existing_values: Optional[Dict[str, float]] = None
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract value with automatic derivation fallback

        Tries extraction in this order:
        1. Direct XBRL tag extraction (from JSON or bulk table)
        2. Derived calculation using formula (if direct extraction fails)

        Args:
            canonical_key: Financial concept to extract
            json_data: SEC CompanyFacts JSON (optional)
            tag_values: Bulk table tag values (optional)
            sector: Company sector
            fiscal_year: Filter to specific fiscal year (JSON only)
            fiscal_period: Filter to specific fiscal period (JSON only)
            existing_values: Already extracted canonical key values for derivation

        Returns:
            Tuple of (value, source) where source is tag name or "derived:<formula>"
        """
        # Try direct extraction first
        if json_data:
            value, tag = self.extract_from_json(
                canonical_key, json_data, sector, fiscal_year, fiscal_period
            )
            if value is not None:
                return (value, tag)

        if tag_values:
            value, tag = self.extract_from_bulk_table(
                canonical_key, tag_values, sector
            )
            if value is not None:
                return (value, tag)

        # Direct extraction failed - try derivation
        if existing_values:
            derived_value = self.calculate_derived_value(canonical_key, existing_values)
            if derived_value is not None:
                formula = self.mappings[canonical_key].get('derived', {}).get('formula', 'unknown')
                return (derived_value, f"derived:{formula}")

        return (None, None)

    def extract_multiple_with_derivation(
        self,
        canonical_keys: List[str],
        json_data: Optional[Dict] = None,
        tag_values: Optional[Dict[str, float]] = None,
        sector: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None
    ) -> Dict[str, Tuple[Optional[float], Optional[str]]]:
        """
        Extract multiple canonical keys with automatic derivation

        Handles dependencies between derived metrics (e.g., extracting total_revenue
        first before deriving cost_of_revenue which depends on it).

        Args:
            canonical_keys: List of financial concepts to extract
            json_data: SEC CompanyFacts JSON (optional)
            tag_values: Bulk table tag values (optional)
            sector: Company sector
            fiscal_year: Filter to specific fiscal year (JSON only)
            fiscal_period: Filter to specific fiscal period (JSON only)

        Returns:
            Dictionary mapping canonical_key -> (value, source)
        """
        results = {}
        values_dict = {}

        # Multi-pass extraction to handle dependencies
        # Pass 1: Extract all directly available values
        for canonical_key in canonical_keys:
            value, source = self.extract_with_derivation(
                canonical_key,
                json_data=json_data,
                tag_values=tag_values,
                sector=sector,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                existing_values=values_dict
            )

            results[canonical_key] = (value, source)
            if value is not None:
                values_dict[canonical_key] = value

        # Pass 2: Try to derive missing values using values from Pass 1
        for canonical_key in canonical_keys:
            if results[canonical_key][0] is None:  # Not extracted in Pass 1
                derived_value = self.calculate_derived_value(canonical_key, values_dict)
                if derived_value is not None:
                    formula = self.mappings.get(canonical_key, {}).get('derived', {}).get('formula', 'unknown')
                    results[canonical_key] = (derived_value, f"derived:{formula}")
                    values_dict[canonical_key] = derived_value

        return results

    def is_metric_excluded(
        self,
        canonical_key: str,
        sector: Optional[str] = None
    ) -> bool:
        """
        Check if a metric should be excluded for a given sector

        Args:
            canonical_key: Financial concept (e.g., 'inventory', 'current_ratio')
            sector: Company sector (e.g., 'Financials', 'Real Estate')

        Returns:
            True if metric is not applicable for this sector, False otherwise
        """
        if not self.sector_mappings or not sector:
            return False

        sector_config = self.sector_mappings.get(sector, {})
        excluded_metrics = sector_config.get('excluded_metrics', [])

        return canonical_key in excluded_metrics

    def is_ratio_excluded(
        self,
        ratio_name: str,
        sector: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[List[str]]]:
        """
        Check if a ratio should be excluded for a given sector

        Args:
            ratio_name: Ratio name (e.g., 'current_ratio', 'quick_ratio')
            sector: Company sector (e.g., 'Financials', 'Energy')

        Returns:
            Tuple of (is_excluded, reason, alternative_metrics)
            - is_excluded: True if ratio not applicable for this sector
            - reason: Explanation why ratio is excluded (or None)
            - alternative_metrics: List of suggested alternatives (or None)
        """
        if not self.sector_mappings or not sector:
            return (False, None, None)

        sector_config = self.sector_mappings.get(sector, {})
        excluded_ratios = sector_config.get('excluded_ratios', {})

        if ratio_name not in excluded_ratios:
            return (False, None, None)

        exclusion_config = excluded_ratios[ratio_name]
        reason = exclusion_config.get('reason')
        alternatives = exclusion_config.get('alternative_metrics', [])

        return (True, reason, alternatives)

    def get_sector_specific_metrics(
        self,
        sector: Optional[str] = None,
        industry: Optional[str] = None
    ) -> List[str]:
        """
        Get sector/industry-specific metrics that should be included in analysis

        Args:
            sector: Company sector (e.g., 'Financials')
            industry: Company industry (e.g., 'Banks', 'Insurance')

        Returns:
            List of XBRL tags for sector/industry-specific metrics
        """
        if not self.sector_mappings or not sector:
            return []

        sector_config = self.sector_mappings.get(sector, {})
        sector_metrics = sector_config.get('sector_specific_metrics', {})

        # Try industry-specific first
        if industry and industry in sector_metrics:
            return sector_metrics[industry]

        # Fall back to sector default
        return sector_metrics.get('_default', [])

    def get_all_tags_for_extraction(
        self,
        canonical_keys: List[str],
        sector: Optional[str] = None
    ) -> List[str]:
        """
        Get unique list of ALL tags needed for a set of canonical keys

        Useful for bulk extraction (single query for all needed tags)

        Args:
            canonical_keys: List of financial concepts to extract
            sector: Company sector

        Returns:
            Deduplicated list of all XBRL tags needed
        """
        all_tags = set()

        for canonical_key in canonical_keys:
            tags = self.get_tags(canonical_key, sector=sector)
            all_tags.update(tags)

        return list(all_tags)

    def get_stats(self) -> Dict:
        """Get extraction statistics"""
        total = self.stats['extractions'] + self.stats['failures']
        success_rate = (self.stats['extractions'] / total * 100) if total > 0 else 0
        fallback_rate = (self.stats['fallbacks_used'] / self.stats['extractions'] * 100) if self.stats['extractions'] > 0 else 0

        return {
            'total_extractions': total,
            'successful': self.stats['extractions'],
            'failed': self.stats['failures'],
            'success_rate': f"{success_rate:.1f}%",
            'fallbacks_used': self.stats['fallbacks_used'],
            'fallback_rate': f"{fallback_rate:.1f}%",
        }


# Singleton instance
_canonical_mapper = None


def get_canonical_mapper(
    mappings_path: Optional[str] = None,
    sector_mappings_path: Optional[str] = None
) -> CanonicalKeyMapper:
    """
    Get singleton CanonicalKeyMapper instance

    Args:
        mappings_path: Path to canonical_key_mappings.json (optional)
        sector_mappings_path: Path to sector_specific_mappings.json (optional)

    Returns:
        Singleton CanonicalKeyMapper instance
    """
    global _canonical_mapper

    if _canonical_mapper is None:
        _canonical_mapper = CanonicalKeyMapper(mappings_path, sector_mappings_path)

    return _canonical_mapper
