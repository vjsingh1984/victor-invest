"""
SEC Data Normalizer with XBRL Tag Mapping

Provides unified extraction from SEC CompanyFacts API data using the XBRL tag
alias mapper for robust cross-company extraction.

Usage:
    from utils.sec_data_normalizer import SECDataNormalizer

    normalizer = SECDataNormalizer()
    metrics = normalizer.extract_from_companyfacts(companyfacts_data, symbol='AAPL')
"""

import logging
from typing import Dict, Optional, Any, List, Tuple
from utils.xbrl_tag_aliases import get_tag_mapper, XBRLTagAliasMapper

logger = logging.getLogger(__name__)


class SECDataNormalizer:
    """
    Normalizes SEC CompanyFacts data to canonical snake_case metrics.

    Handles:
    1. XBRL tag variations across companies (via XBRLTagAliasMapper)
    2. Fiscal period extraction
    3. Unit conversion (USD preferred)
    4. Latest value extraction (annual vs quarterly)
    """

    def __init__(self):
        """Initialize with XBRL tag mapper."""
        self.mapper: XBRLTagAliasMapper = get_tag_mapper()

    def extract_from_companyfacts(
        self,
        facts_data: Dict[str, Any],
        symbol: str,
        prefer_annual: bool = True
    ) -> Dict[str, Any]:
        """
        Extract financial metrics from SEC CompanyFacts API response.

        Args:
            facts_data: CompanyFacts API response dict
            symbol: Stock ticker symbol
            prefer_annual: Prefer annual values over quarterly (default: True)

        Returns:
            Dict with canonical snake_case keys and extracted values

        Example:
            >>> facts = fetch_sec_api('0000320193')  # AAPL
            >>> metrics = normalizer.extract_from_companyfacts(facts, 'AAPL')
            >>> print(metrics['total_revenue'])  # Works regardless of tag variation
        """
        if not facts_data or 'facts' not in facts_data:
            logger.warning(f"No facts data found for {symbol}")
            return self._empty_metrics(symbol)

        # Check for flattened cache (already processed data)
        us_gaap = facts_data['facts'].get('us-gaap', {})
        if not us_gaap:
            logger.debug(f"Detected flattened/processed cache for {symbol}")
            return {
                **facts_data.get('facts', {}),
                'symbol': symbol
            }

        # Extract metrics using tag mapper
        metrics = {}

        # Get fiscal period info from revenue (most reliable)
        fiscal_year, fiscal_period = self._extract_fiscal_period(us_gaap, symbol)
        metrics['fiscal_year'] = fiscal_year
        metrics['fiscal_period'] = fiscal_period
        metrics['symbol'] = symbol

        # Extract all canonical metrics
        for canonical_name in self.mapper.get_all_canonical_names():
            value = self._extract_metric_with_fallback(
                us_gaap,
                canonical_name,
                prefer_annual
            )
            if value is not None:
                metrics[canonical_name] = value

        # Log coverage
        coverage_stats = self._calculate_coverage(metrics)
        logger.info(
            f"SEC data extraction for {symbol}: "
            f"{coverage_stats['found']}/{coverage_stats['total']} metrics "
            f"({coverage_stats['pct']:.1f}% coverage)"
        )

        if coverage_stats['pct'] < 50:
            logger.warning(
                f"Low metric coverage for {symbol} ({coverage_stats['pct']:.1f}%). "
                f"Missing: {coverage_stats['missing'][:5]}..."
            )

        return metrics

    def _extract_metric_with_fallback(
        self,
        us_gaap: Dict[str, Any],
        canonical_name: str,
        prefer_annual: bool = True
    ) -> Optional[float]:
        """
        Extract a metric using XBRL tag fallback logic.

        Args:
            us_gaap: US-GAAP section from CompanyFacts
            canonical_name: Canonical metric name (e.g., 'total_revenue')
            prefer_annual: Prefer annual values over quarterly

        Returns:
            Extracted value or None if not found
        """
        # Get all possible XBRL tags for this metric
        xbrl_tags = self.mapper.get_xbrl_aliases(canonical_name)

        if not xbrl_tags:
            # Metric not in our mapping system
            return None

        # Try each XBRL tag in priority order
        for idx, xbrl_tag in enumerate(xbrl_tags):
            concept = us_gaap.get(xbrl_tag, {})
            if not concept:
                continue

            units = concept.get('units', {})

            # Try USD first (preferred)
            usd_data = units.get('USD', [])
            if usd_data:
                value = self._get_latest_value(usd_data, prefer_annual)
                if value is not None:
                    if idx > 0:
                        # Log when fallback is used
                        logger.debug(
                            f"Extracted '{canonical_name}' using fallback tag '{xbrl_tag}' "
                            f"(priority {idx + 1}/{len(xbrl_tags)})"
                        )
                    return value

            # Try other currency units if USD not available
            for unit_name, unit_data in units.items():
                if unit_data and unit_name != 'USD':
                    value = self._get_latest_value(unit_data, prefer_annual)
                    if value is not None:
                        logger.debug(
                            f"Extracted '{canonical_name}' from non-USD unit '{unit_name}' "
                            f"using tag '{xbrl_tag}'"
                        )
                        return value

        # No value found after trying all aliases
        logger.debug(
            f"Could not extract '{canonical_name}' "
            f"(tried {len(xbrl_tags)} XBRL tags: {xbrl_tags[:3]}...)"
        )
        return None

    def _get_latest_value(
        self,
        data: List[Dict],
        prefer_annual: bool = True
    ) -> Optional[float]:
        """
        Extract the latest value from XBRL data points.

        Args:
            data: List of XBRL data points (from CompanyFacts API)
            prefer_annual: Prefer annual values (fy) over quarterly (default: True)

        Returns:
            Latest value or None if not found
        """
        if not data:
            return None

        # Filter for annual or quarterly data
        if prefer_annual:
            # Prefer FY (annual) data
            fy_data = [d for d in data if d.get('fp') == 'FY' and d.get('form') in ('10-K', '10-K/A')]
            if fy_data:
                # Sort by end date (descending) and get latest
                sorted_data = sorted(fy_data, key=lambda x: x.get('end', ''), reverse=True)
                return sorted_data[0].get('val')

        # Fallback to latest value regardless of period
        sorted_data = sorted(data, key=lambda x: x.get('end', ''), reverse=True)
        return sorted_data[0].get('val') if sorted_data else None

    def _get_latest_value_with_period(
        self,
        data: List[Dict]
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        """
        Extract latest value along with fiscal year and period.

        Args:
            data: List of XBRL data points

        Returns:
            Tuple of (value, fiscal_year, fiscal_period) or (None, None, None)
        """
        if not data:
            return (None, None, None)

        # Sort by end date (descending)
        sorted_data = sorted(data, key=lambda x: x.get('end', ''), reverse=True)

        if sorted_data:
            latest = sorted_data[0]
            return (
                latest.get('val'),
                latest.get('fy'),
                latest.get('fp')
            )

        return (None, None, None)

    def _extract_fiscal_period(
        self,
        us_gaap: Dict[str, Any],
        symbol: str
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Extract fiscal year and period from revenue data.

        Args:
            us_gaap: US-GAAP section from CompanyFacts
            symbol: Stock ticker (for logging)

        Returns:
            Tuple of (fiscal_year, fiscal_period) or (None, None)
        """
        # Try revenue-related tags in priority order
        revenue_tags = self.mapper.get_xbrl_aliases('total_revenue')

        for tag in revenue_tags:
            concept = us_gaap.get(tag, {})
            units = concept.get('units', {})
            usd_data = units.get('USD', [])

            if usd_data:
                _, fiscal_year, fiscal_period = self._get_latest_value_with_period(usd_data)
                if fiscal_year and fiscal_period:
                    logger.debug(
                        f"Extracted fiscal period for {symbol} from '{tag}': "
                        f"{fiscal_year}-{fiscal_period}"
                    )
                    return (fiscal_year, fiscal_period)

        logger.warning(f"Could not extract fiscal period for {symbol}")
        return (None, None)

    def _empty_metrics(self, symbol: str) -> Dict[str, Any]:
        """Return empty metrics dict with symbol."""
        return {
            'symbol': symbol,
            'fiscal_year': None,
            'fiscal_period': None,
        }

    def _calculate_coverage(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metric coverage statistics.

        Args:
            metrics: Extracted metrics dict

        Returns:
            Dict with coverage stats
        """
        # Exclude metadata fields
        metadata_fields = {'symbol', 'fiscal_year', 'fiscal_period'}

        all_canonical = set(self.mapper.get_all_canonical_names())
        extracted = set(metrics.keys()) - metadata_fields
        found = all_canonical & extracted
        missing = all_canonical - found

        return {
            'total': len(all_canonical),
            'found': len(found),
            'missing': list(missing),
            'pct': (len(found) / len(all_canonical) * 100) if all_canonical else 0
        }

    def normalize_bulk_table_row(
        self,
        row: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Normalize a single row from SEC bulk tables (sec_num_data).

        Converts XBRL tag → canonical name using mapper.

        Args:
            row: Row dict with 'tag' and 'value' keys
            symbol: Stock ticker

        Returns:
            Dict with canonical name as key

        Example:
            >>> row = {'tag': 'Revenues', 'value': 100_000_000}
            >>> normalized = normalizer.normalize_bulk_table_row(row, 'NFLX')
            >>> print(normalized)  # {'total_revenue': 100_000_000}
        """
        xbrl_tag = row.get('tag')
        value = row.get('value')

        if not xbrl_tag or value is None:
            return {}

        # Resolve XBRL tag to canonical name
        canonical = self.mapper.resolve_to_canonical(xbrl_tag)

        if canonical:
            return {canonical: value}
        else:
            # Unknown tag - log and return as-is
            logger.debug(f"Unknown XBRL tag for {symbol}: {xbrl_tag} (not in mapper)")
            return {xbrl_tag: value}  # Keep original for debugging

    def normalize_bulk_table_results(
        self,
        rows: List[Dict[str, Any]],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Normalize multiple rows from SEC bulk tables.

        Args:
            rows: List of row dicts from sec_num_data table
            symbol: Stock ticker

        Returns:
            Dict with canonical keys

        Example:
            >>> rows = [
            ...     {'tag': 'Revenues', 'value': 100_000_000},
            ...     {'tag': 'NetIncomeLoss', 'value': 20_000_000},
            ... ]
            >>> metrics = normalizer.normalize_bulk_table_results(rows, 'NFLX')
            >>> print(metrics['total_revenue'])  # 100_000_000
        """
        normalized = {'symbol': symbol}

        for row in rows:
            row_normalized = self.normalize_bulk_table_row(row, symbol)
            # Only set canonical keys if not already set (first alias wins, not last)
            for key, value in row_normalized.items():
                if key not in normalized:
                    normalized[key] = value

        # Log coverage
        coverage_stats = self._calculate_coverage(normalized)
        logger.info(
            f"Bulk table extraction for {symbol}: "
            f"{coverage_stats['found']}/{coverage_stats['total']} metrics "
            f"({coverage_stats['pct']:.1f}% coverage)"
        )

        return normalized


# Convenience function for quick access
def extract_sec_metrics(
    facts_data: Dict[str, Any],
    symbol: str,
    prefer_annual: bool = True
) -> Dict[str, Any]:
    """
    Quick extraction of SEC metrics using tag mapper.

    Args:
        facts_data: CompanyFacts API response
        symbol: Stock ticker
        prefer_annual: Prefer annual values

    Returns:
        Dict with canonical metrics
    """
    normalizer = SECDataNormalizer()
    return normalizer.extract_from_companyfacts(facts_data, symbol, prefer_annual)
