#!/usr/bin/env python3
"""
Quarterly Metrics Calculator - Clean Architecture Migration

CANONICAL LOCATION: investigator.domain.services.quarterly_processor
MIGRATION DATE: 2025-11-17

Computes implicit/missing quarterly data from FY and reported quarters.
Ensures TTM calculations use the most recent N consecutive quarters.

Key Logic:
- FY = Q1 + Q2 + Q3 + Q4 (cumulative annual data)
- If we have FY, Q1, Q2, Q3 → compute Q4 = FY - (Q1 + Q2 + Q3)
- Always use rolling N most recent quarters for TTM (default 4, configurable to 8, 12, 16)

═══════════════════════════════════════════════════════════════════════════════
CRITICAL: FY Period Label Handling Strategy
═══════════════════════════════════════════════════════════════════════════════

**WHY WE NEED FY PERIODS:**
SEC companies report quarterly (Q1, Q2, Q3) and annual (FY) data. However, Q4 is
RARELY reported separately - it must be computed from: Q4 = FY - (Q1 + Q2 + Q3)

**FY PERIOD LIFECYCLE:**

1. INPUT (Database/API Fetch):
   - FY periods MUST be fetched alongside Q periods
   - fiscal_period='FY', values are Year-To-Date (YTD) cumulative totals
   - Example: ORCL FY ending 2025-05-31 contains sum of Q1+Q2+Q3+Q4

2. PROCESSING (This Module):
   - compute_missing_quarter() takes FY period data as input
   - Subtracts Q1+Q2+Q3 from FY to derive Q4
   - Returns NEW period with fiscal_period='Q4', values=Point-In-Time (PIT)
   - Original FY period is NOT modified

3. OUTPUT (Caller's Responsibility):
   - AFTER Q4 computation, FY periods MUST be filtered out
   - Final output should contain ONLY Q periods (Q1, Q2, Q3, Q4)
   - FY and Q4 periods MUST NEVER appear together in final output
   - Reason: FY contains YTD cumulative, Q4 is PIT individual quarter
   - Using both would double-count Q4 data in TTM calculations

**CORRECT DATA FLOW:**
```
Database Query: [FY-2025, Q3-2025, Q2-2025, Q1-2025, FY-2024, ...]
                  ↓
compute_missing_quarter(): FY-2025 → Q4-2025 (fiscal_period='Q4', is_ytd=False)
                  ↓
Filter FY Periods: [Q4-2025, Q3-2025, Q2-2025, Q1-2025, Q4-2024, ...]
                  ↓
DCF/TTM Calculations: Use only Q periods (NO FY periods present)
```

**WHY Q4 IS LABELED 'Q4' (NOT 'FY'):**
- Computed values are Point-In-Time (PIT), representing ONLY Q4 activity
- is_ytd=False explicitly marks them as individual quarter, not cumulative
- fiscal_period='Q4' prevents confusion with original FY YTD data
- Allows proper sorting: Q1 → Q2 → Q3 → Q4 (chronological order)
- TTM calculations expect Q periods, not FY periods

**EDGE CASES:**
- Missing Q1/Q2: Can still compute Q4 if Q3 is YTD (Q4 = FY - Q3_YTD)
- No Q3: Cannot compute Q4 reliably (need at least 2 quarters or YTD Q3)
- Q4 already reported: Skip computation (rare, but some companies do report Q4)

═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Import FiscalPeriodService for centralized fiscal period handling
from investigator.domain.services.fiscal_period_service import get_fiscal_period_service

logger = logging.getLogger(__name__)

# Module-level cache to suppress duplicate YTD conversion warnings
# Key format: "Q{period}-{fiscal_year}-{warning_type}"
_ytd_warnings_logged = set()


def compute_missing_quarter(
    fy_data: Dict[str, Any],
    q1_data: Optional[Dict[str, Any]] = None,
    q2_data: Optional[Dict[str, Any]] = None,
    q3_data: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Compute Q4 metrics implicitly from FY and Q1-Q3.

    ═══════════════════════════════════════════════════════════════════════════
    CRITICAL: FY Period Label Handling
    ═══════════════════════════════════════════════════════════════════════════

    **INPUT:** Takes FY period data (fiscal_period='FY', values=YTD cumulative)
    **OUTPUT:** Returns Q4 period data (fiscal_period='Q4', values=PIT individual quarter)

    **FY PERIODS SHOULD BE:**
    1. KEPT as input for this computation (needed for formula: Q4 = FY - Q1 - Q2 - Q3)
    2. FILTERED OUT after Q4 computation (caller's responsibility - see fundamental agent)
    3. NEVER appear alongside Q4 in final output (would double-count Q4 data)

    **THE COMPUTED Q4 VALUES ARE POINT-IN-TIME (PIT), NOT YTD:**
    - 'is_ytd': False is explicitly set (lines 171-172 below)
    - Values represent only Q4 activity, not cumulative
    - Example: Q4 revenue = $5.2B (individual quarter), NOT $24.8B (YTD)

    **WHY Q4 IS LABELED 'Q4' (NOT 'FY'):**
    - Prevents confusion with original FY YTD data
    - Allows proper chronological sorting (Q1 → Q2 → Q3 → Q4)
    - TTM calculations expect Q periods, not FY periods
    - Explicit 'computed': True flag marks it as derived data

    ═══════════════════════════════════════════════════════════════════════════

    Formula: Q4 = FY - (Q1 + Q2 + Q3)

    Args:
        fy_data: Full year data dict (fiscal_period='FY', YTD cumulative values)
        q1_data: Q1 data dict (optional)
        q2_data: Q2 data dict (optional)
        q3_data: Q3 data dict (optional)

    Returns:
        Dict with computed Q4 metrics (fiscal_period='Q4', is_ytd=False, computed=True)
        Returns None if insufficient data

    Example:
        >>> fy = {'operating_cash_flow': 110543e6, 'capital_expenditures': 10959e6}
        >>> q1 = {'operating_cash_flow': 39895e6, 'capital_expenditures': 2392e6}
        >>> q2 = {'operating_cash_flow': 62585e6, 'capital_expenditures': 4388e6}
        >>> q3 = {'operating_cash_flow': 91443e6, 'capital_expenditures': 6539e6}
        >>> q4 = compute_missing_quarter(fy, q1, q2, q3)
        >>> # Q4 OCF = 110543 - (39895 + 62585 + 91443) = -83380 (ERROR - data issue!)
    """
    if not fy_data:
        return None

    # Need at least 2 quarters to compute the missing one
    # Exception: If only Q3 is available and it's YTD, we can compute Q4 = FY - Q3_YTD
    available_quarters = [q for q in [q1_data, q2_data, q3_data] if q is not None]

    if len(available_quarters) < 2:
        # Special case: Q3 only with YTD data allows Q4 = FY - Q3_YTD
        if len(available_quarters) == 1 and q3_data is not None:
            q3_is_ytd = (q3_data.get('income_statement', {}).get('is_ytd') or
                        q3_data.get('cash_flow', {}).get('is_ytd'))
            if q3_is_ytd:
                logger.info(
                    f"✅ Q4 computation ALLOWED with only Q3 (YTD): "
                    f"Q1/Q2 missing, but Q3 is YTD, so Q4 = FY - Q3_YTD is valid"
                )
                # Continue to computation
            else:
                logger.debug("Insufficient quarterly data: only Q3 available and not YTD (need at least 2 quarters or YTD Q3)")
                return None
        else:
            logger.debug(f"Insufficient quarterly data to compute missing quarter (need at least 2, have {len(available_quarters)})")
            return None

    # Get fiscal year and period from FY data
    fiscal_year = fy_data.get('fiscal_year')
    if not fiscal_year:
        logger.warning("FY data missing fiscal_year field")
        return None

    def _ensure_free_cash_flow(container: Optional[Dict[str, Any]], label: str) -> None:
        """Derive free cash flow from operating cash flow and capex when missing or zero."""
        if not container:
            return

        def derive(target: Dict[str, Any], prefix: str = "") -> None:
            ocf = target.get('operating_cash_flow')
            capex = target.get('capital_expenditures')
            if ocf is None or capex is None:
                return

            # CRITICAL FIX: Derive FCF if missing OR if explicit zero (database artifact)
            # Only skip derivation if FCF is present AND non-zero
            existing_fcf = target.get('free_cash_flow')
            if existing_fcf and existing_fcf != 0:  # Has non-zero FCF, don't override
                return

            # Derive FCF = OCF - |CapEx|
            derived = ocf - abs(capex)
            target['free_cash_flow'] = derived
            replaced_zero = " [replaced explicit zero]" if existing_fcf == 0 else ""
            logger.debug(
                "   ↳ Derived free_cash_flow for %s%s: %.1fM (OCF %.1fM - |CapEx| %.1fM)%s",
                label,
                prefix,
                derived / 1e6,
                ocf / 1e6,
                abs(capex) / 1e6,
                replaced_zero
            )

        if 'cash_flow' in container and isinstance(container['cash_flow'], dict):
            derive(container['cash_flow'])
        else:
            derive(container)

    # Derive missing FCF for FY and reported quarters before computing differences
    _ensure_free_cash_flow(fy_data, "FY")
    _ensure_free_cash_flow(q1_data, "Q1")
    _ensure_free_cash_flow(q2_data, "Q2")
    _ensure_free_cash_flow(q3_data, "Q3")

    # CRITICAL: Validate that Q4 can be computed
    # Special case: If Q1 and Q2 are missing, but Q3 is YTD, we can compute Q4 = FY - Q3_YTD
    # This handles cases where early quarters aren't filed yet (e.g., 2024-Q1, 2025-Q1)
    q1_is_ytd = q1_data and (q1_data.get('income_statement', {}).get('is_ytd') or q1_data.get('cash_flow', {}).get('is_ytd'))
    q2_is_ytd = q2_data and (q2_data.get('income_statement', {}).get('is_ytd') or q2_data.get('cash_flow', {}).get('is_ytd'))
    q3_is_ytd = q3_data and (q3_data.get('income_statement', {}).get('is_ytd') or q3_data.get('cash_flow', {}).get('is_ytd'))

    # If Q3 is YTD and Q1/Q2 are missing or also YTD, we can compute Q4 = FY - Q3_YTD
    if q3_is_ytd and not q1_data and not q2_data:
        logger.info(
            f"✅ Q4 computation ALLOWED for FY {fiscal_year}: Q1/Q2 missing, Q3 is YTD. "
            f"Will compute Q4 = FY - Q3_YTD (valid SEC calculation)."
        )
        # Continue with Q4 computation using FY - Q3_YTD
    elif q1_is_ytd or q2_is_ytd or (q3_is_ytd and (q1_data or q2_data)):
        # YTD data detected in Q1/Q2, or Q3 with Q1/Q2 present means conversion should have happened
        ytd_detected = []
        if q1_is_ytd:
            ytd_detected.append("Q1")
        if q2_is_ytd:
            ytd_detected.append("Q2")
        if q3_is_ytd and (q1_data or q2_data):
            ytd_detected.append("Q3")

        # Use cache to suppress duplicate warnings
        warning_key = f"Q4-{fiscal_year}-ytd_data_detected-{','.join(ytd_detected)}"
        if warning_key not in _ytd_warnings_logged:
            logger.warning(
                f"⚠️  Q4 computation SKIPPED for FY {fiscal_year}: YTD data detected in {', '.join(ytd_detected)}. "
                f"This indicates convert_ytd_to_quarterly() was not called or failed. "
                f"YTD conversion requires previous quarters to be present for subtraction."
            )
            _ytd_warnings_logged.add(warning_key)
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # Initialize Q4 dict with statement-level structure
    # CRITICAL: fiscal_period='Q4' (NOT 'FY') and is_ytd=False (PIT, not cumulative)
    # ═══════════════════════════════════════════════════════════════════════════
    q4_computed = {
        'symbol': fy_data.get('symbol'),
        'fiscal_year': fiscal_year,
        'fiscal_period': 'Q4',  # ← LABELED AS Q4 (not FY) - represents individual quarter
        'computed': True,  # Flag to indicate this is derived data
        'computation_method': 'FY_minus_reported_quarters',
        'cash_flow': {'is_ytd': False},  # ← Q4 is POINT-IN-TIME, not YTD cumulative
        'income_statement': {'is_ytd': False},  # ← Q4 is POINT-IN-TIME, not YTD cumulative
        'balance_sheet': {},  # Will be populated from FY data
        'ratios': {}  # Will be populated from FY data
    }

    # CLEAN ARCHITECTURE: Statement-level structure
    # Cash flow metrics
    cash_flow_keys = [
        'operating_cash_flow',
        'capital_expenditures',
        'free_cash_flow',
        'financing_cash_flow',
        'investing_cash_flow',
        'dividends_paid'
    ]

    # Income statement metrics
    income_keys = [
        'total_revenue',
        'net_income',
        'operating_income',
        'gross_profit',
        'cost_of_revenue'
    ]

    # DEBUG: Log input data structure for debugging
    logger.info(f"🔧 [Q4_COMPUTE] Computing Q4 for FY {fiscal_year}")
    logger.info(f"   FY data structure: cash_flow={('cash_flow' in fy_data)}, income_statement={('income_statement' in fy_data)}")
    logger.info(f"   Available quarters: {len(available_quarters)} quarters")
    for i, q in enumerate(available_quarters):
        fp = q.get('fiscal_period', 'Unknown')
        logger.info(f"   Q{i+1} ({fp}): cash_flow={('cash_flow' in q)}, income_statement={('income_statement' in q)}")

    # Compute cash flow metrics for Q4
    for key in cash_flow_keys:
        # Get FY value
        fy_value = None
        if 'cash_flow' in fy_data and key in fy_data['cash_flow']:
            fy_value = fy_data['cash_flow'].get(key)
        elif key in fy_data:  # Fallback for flat structure
            fy_value = fy_data.get(key)

        if fy_value is None:
            logger.debug(f"   Skipping {key}: FY value is None")
            continue

        logger.debug(f"   Processing {key}: FY={fy_value/1e6:.1f}M")

        # Sum values from reported quarters
        quarters_sum = 0
        quarters_count = 0
        for q_data in available_quarters:
            q_value = None
            if 'cash_flow' in q_data and key in q_data['cash_flow']:
                q_value = q_data['cash_flow'].get(key)
            elif key in q_data:  # Fallback for flat structure
                q_value = q_data.get(key)

            if q_value is not None:
                fp = q_data.get('fiscal_period', '??')
                logger.debug(f"      {fp}: {key}={q_value/1e6:.1f}M")
                quarters_sum += q_value
                quarters_count += 1

        # Compute Q4 value
        # Allow with 1 quarter if Q3 is YTD (special case for missing Q1/Q2)
        q3_is_ytd = q3_data and (q3_data.get('income_statement', {}).get('is_ytd') or q3_data.get('cash_flow', {}).get('is_ytd'))
        min_quarters_needed = 1 if (q3_is_ytd and not q1_data and not q2_data) else 2

        if quarters_count >= min_quarters_needed:
            q4_value = fy_value - quarters_sum
            q4_computed['cash_flow'][key] = q4_value
            logger.info(f"   ✅ {key}: Q4={q4_value/1e6:.1f}M (FY {fy_value/1e6:.1f}M - Q1-Q3 {quarters_sum/1e6:.1f}M)")

            # Validation: Q4 should be reasonable (not negative for most metrics except capex)
            if key not in ['capital_expenditures', 'dividends_paid'] and q4_value < 0:
                logger.warning(
                    f"Computed negative Q4 {key}: {q4_value/1e6:.1f}M "
                    f"(FY: {fy_value/1e6:.1f}M, Q1-Q3 sum: {quarters_sum/1e6:.1f}M). "
                    f"This may indicate data quality issues."
                )

    # If free cash flow wasn't explicitly computed but we have OCF and CapEx, derive it now
    q4_cf = q4_computed.get('cash_flow', {})
    if (
        'free_cash_flow' not in q4_cf
        and q4_cf.get('operating_cash_flow') is not None
        and q4_cf.get('capital_expenditures') is not None
    ):
        ocf = q4_cf['operating_cash_flow']
        capex = q4_cf['capital_expenditures']
        derived_fcf = ocf - abs(capex)
        q4_cf['free_cash_flow'] = derived_fcf
        logger.info(
            "   🔁 free_cash_flow derived from operating_cash_flow %.1fM - |capital_expenditures| %.1fM = %.1fM",
            ocf / 1e6,
            abs(capex) / 1e6,
            derived_fcf / 1e6,
        )

    # Compute income statement metrics for Q4
    for key in income_keys:
        # Get FY value
        fy_value = None
        if 'income_statement' in fy_data and key in fy_data['income_statement']:
            fy_value = fy_data['income_statement'].get(key)
        elif key in fy_data:  # Fallback for flat structure
            fy_value = fy_data.get(key)

        if fy_value is None:
            continue

        # Sum values from reported quarters
        quarters_sum = 0
        quarters_count = 0
        for q_data in available_quarters:
            q_value = None
            if 'income_statement' in q_data and key in q_data['income_statement']:
                q_value = q_data['income_statement'].get(key)
            elif key in q_data:  # Fallback for flat structure
                q_value = q_data.get(key)

            if q_value is not None:
                quarters_sum += q_value
                quarters_count += 1

        # Compute Q4 value
        # Allow with 1 quarter if Q3 is YTD (special case for missing Q1/Q2)
        q3_is_ytd = q3_data and (q3_data.get('income_statement', {}).get('is_ytd') or q3_data.get('cash_flow', {}).get('is_ytd'))
        min_quarters_needed = 1 if (q3_is_ytd and not q1_data and not q2_data) else 2

        if quarters_count >= min_quarters_needed:
            q4_value = fy_value - quarters_sum
            q4_computed['income_statement'][key] = q4_value

            # Validation: Q4 should be reasonable (not negative)
            if q4_value < 0 and key not in ['net_income', 'operating_income']:
                logger.warning(
                    f"Computed negative Q4 {key}: {q4_value/1e6:.1f}M "
                    f"(FY: {fy_value/1e6:.1f}M, Q1-Q3 sum: {quarters_sum/1e6:.1f}M). "
                    f"This may indicate data quality issues."
                )

    # Copy balance sheet from FY (it's a point-in-time snapshot)
    if 'balance_sheet' in fy_data:
        q4_computed['balance_sheet'] = fy_data['balance_sheet'].copy()

    # Copy ratios from FY (or could recompute from Q4 metrics)
    if 'ratios' in fy_data:
        q4_computed['ratios'] = fy_data['ratios'].copy()

    # Check if we computed at least some metrics
    has_cash_flow = len(q4_computed['cash_flow']) > 1  # More than just is_ytd
    has_income = len(q4_computed['income_statement']) > 1  # More than just is_ytd

    return q4_computed if (has_cash_flow or has_income) else None


def extract_nested_value(data_dict: Dict[str, Any], key: str, debug: bool = False) -> Optional[float]:
    """
    Extract value from dict, handling nested financial_data structure and dot notation.

    Supports:
    1. Dot notation: 'cash_flow.free_cash_flow' → data_dict['cash_flow']['free_cash_flow']
    2. Direct access: data_dict[key]
    3. Nested: data_dict['financial_data']['cash_flow_statement'][key]
    4. Nested: data_dict['financial_data']['income_statement'][key]

    Args:
        data_dict: Dictionary to search
        key: Key to find (may include dots for nested paths)
        debug: Enable debug logging

    Returns:
        Float value or None
    """
    if debug:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 extract: key='{key}', top_keys={list(data_dict.keys())}")

    # FIRST: Handle dot notation (e.g., 'cash_flow.free_cash_flow')
    if '.' in key:
        parts = key.split('.')
        current = data_dict
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                if debug:
                    logger.warning(f"🔍 extract: Path navigation failed at '{part}' in key='{key}'")
                return None

        if isinstance(current, (int, float)):
            if debug:
                logger.info(f"🔍 extract: FOUND via dot notation key='{key}', val={current}")
            return float(current)
        elif current is None:
            return None
        else:
            if debug:
                logger.warning(f"🔍 extract: Dot notation resulted in non-numeric value: type={type(current)}")
            return None

    # Try direct access
    if key in data_dict:
        val = data_dict[key]
        if debug:
            logger.info(f"🔍 extract: FOUND DIRECT key='{key}', val={val}")
        return float(val) if val is not None else None

    # Try nested in financial_data
    if 'financial_data' in data_dict:
        fd = data_dict['financial_data']
        if not isinstance(fd, dict):
            if debug:
                logger.warning(f"🔍 extract: financial_data NOT dict, type={type(fd)}")
            return None

        if debug:
            logger.info(f"🔍 extract: financial_data keys={list(fd.keys())}")

        # FIRST: Try directly in financial_data (flat structure from sec_companyfacts_processed)
        if key in fd:
            val = fd[key]
            if debug:
                logger.info(f"🔍 extract: FOUND in FINANCIAL_DATA key='{key}', val={val}")
            return float(val) if val is not None else None

        # Try cash_flow_statement
        if 'cash_flow_statement' in fd and isinstance(fd['cash_flow_statement'], dict):
            if debug:
                logger.info(f"🔍 extract: cash_flow keys={list(fd['cash_flow_statement'].keys())[:5]}")
            if key in fd['cash_flow_statement']:
                val = fd['cash_flow_statement'][key]
                if debug:
                    logger.info(f"🔍 extract: FOUND in CASH_FLOW key='{key}', val={val}")
                return float(val) if val is not None else None

        # Try income_statement
        if 'income_statement' in fd and isinstance(fd['income_statement'], dict):
            if key in fd['income_statement']:
                val = fd['income_statement'][key]
                if debug:
                    logger.info(f"🔍 extract: FOUND in INCOME key='{key}', val={val}")
                return float(val) if val is not None else None

        # Try balance_sheet
        if 'balance_sheet' in fd and isinstance(fd['balance_sheet'], dict):
            if key in fd['balance_sheet']:
                val = fd['balance_sheet'][key]
                if debug:
                    logger.info(f"🔍 extract: FOUND in BALANCE key='{key}', val={val}")
                return float(val) if val is not None else None

    if debug:
        logger.warning(f"🔍 extract: NOT FOUND key='{key}'")
    return None


def convert_ytd_to_quarterly(quarters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert YTD (Year-To-Date) cumulative values to individual quarter values.

    SEC 10-Q filings report YTD cumulative values for Q2/Q3:
    - Q1: Individual quarter
    - Q2: YTD (Q1+Q2) → need to subtract Q1
    - Q3: YTD (Q1+Q2+Q3) → need to subtract Q1+Q2
    - Q4: Individual quarter (computed from FY - Q1 - Q2 - Q3)

    Args:
        quarters: List of quarterly periods sorted by fiscal_year/fiscal_period descending

    Returns:
        List of quarters with YTD values converted to individual quarter values

    Example:
        Q2 YTD = $53.89B → Q2 individual = $53.89B - $29.94B (Q1) = $23.95B
        Q3 YTD = $81.75B → Q3 individual = $81.75B - $53.89B (Q2 YTD) = $27.86B
    """
    if not quarters:
        return []

    # Group quarters by fiscal year
    # CRITICAL FIX: Use fiscal_year label to create separate groups, not proximity
    # Previous bug: All quarters within 365 days went into ONE group, causing overwrites
    # Example: Q3-2025, Q3-2024, Q3-2023 all overwrote the 'Q3' key in the same dict
    fiscal_year_groups: List[Dict[str, Dict]] = []

    def parse_date(q):
        date_str = q.get('period_end_date', '')
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return None

    # Group by fiscal_year label first, then by period
    # This ensures Q1-2025, Q2-2025, Q3-2025 stay together
    fy_dict: Dict[int, Dict[str, Dict]] = {}

    for q in quarters:
        period = q.get('fiscal_period', '')
        if not period.startswith('Q'):
            continue

        fiscal_year = q.get('fiscal_year')
        if not fiscal_year:
            logger.warning(f"Quarter missing fiscal_year: {q.get('period_end_date')}, skipping YTD conversion")
            continue

        # Create fiscal year group if doesn't exist
        if fiscal_year not in fy_dict:
            fy_dict[fiscal_year] = {}

        # Add quarter to its fiscal year group (indexed by period: Q1, Q2, Q3)
        fy_dict[fiscal_year][period] = q

        logger.debug(
            f"[YTD_GROUP] Added {period}-{fiscal_year} (ending {q.get('period_end_date')}) to FY{fiscal_year} group"
        )

    # Convert dict to list of groups for backward compatibility
    fiscal_year_groups = list(fy_dict.values())

    logger.info(
        f"[YTD_GROUP] Created {len(fiscal_year_groups)} fiscal year groups: "
        f"{[(fy, list(group.keys())) for fy, group in fy_dict.items()]}"
    )

    # Process each fiscal year group
    for year_quarters in fiscal_year_groups:
        # CRITICAL DATA QUALITY CHECK: Detect missing quarters before YTD conversion
        # Missing quarters → Failed YTD conversion → Corrupt negative revenue
        # Examples: ORCL Q2 (-$1,085M), ZS Q1 (NULL), META Q2 (-$178M)

        # Convert Q2 if marked as YTD
        if 'Q2' in year_quarters:
            q2 = year_quarters['Q2']

            # Check if Q2 is YTD and requires conversion
            if q2.get('income_statement', {}).get('is_ytd') or q2.get('cash_flow', {}).get('is_ytd'):
                # CRITICAL: Q2 YTD conversion requires Q1
                if 'Q1' not in year_quarters:
                    fiscal_year = q2.get('fiscal_year', 'Unknown')
                    # Use cache to suppress duplicate warnings
                    warning_key = f"Q2-{fiscal_year}-missing_q1"
                    if warning_key not in _ytd_warnings_logged:
                        logger.warning(
                            f"⚠️  YTD CONVERSION SKIPPED: Q2-{fiscal_year} is YTD but Q1 is missing from dataset. "
                            f"Cannot convert YTD to quarterly without Q1. This is expected if company didn't file Q1 or it's outside data range. "
                            f"Marking quarter as conversion_failed."
                        )
                        _ytd_warnings_logged.add(warning_key)
                    # Mark as conversion_failed to prevent downstream use
                    q2['ytd_conversion_failed'] = True
                    q2['ytd_conversion_error'] = 'Missing Q1'
                    continue  # Skip this Q2 conversion

            q1 = year_quarters.get('Q1')
            if q1:
                # Check if income_statement is YTD
                if q2.get('income_statement', {}).get('is_ytd'):
                    income = q2['income_statement']
                    q1_income = q1.get('income_statement', {})

                    # Subtract Q1 from Q2 YTD to get Q2 individual
                    for key in income:
                        if key != 'is_ytd' and isinstance(income.get(key), (int, float)):
                            q1_val = q1_income.get(key, 0) or 0
                            q2_ytd_val = income[key] or 0
                            income[key] = q2_ytd_val - q1_val

                    income['is_ytd'] = False
                    q2_end_date = q2.get('period_end_date', 'Unknown')
                    logger.debug(f"Converted Q2 ending {q2_end_date} income_statement from YTD to individual quarter")

                # Check if cash_flow is YTD
                if q2.get('cash_flow', {}).get('is_ytd'):
                    cash_flow = q2['cash_flow']
                    q1_cash_flow = q1.get('cash_flow', {})

                    # Subtract Q1 from Q2 YTD to get Q2 individual
                    for key in cash_flow:
                        if key != 'is_ytd' and isinstance(cash_flow.get(key), (int, float)):
                            q1_val = q1_cash_flow.get(key, 0) or 0
                            q2_ytd_val = cash_flow[key] or 0
                            cash_flow[key] = q2_ytd_val - q1_val

                    cash_flow['is_ytd'] = False
                    q2_end_date = q2.get('period_end_date', 'Unknown')
                    logger.debug(f"Converted Q2 ending {q2_end_date} cash_flow from YTD to individual quarter")

        # Convert Q3 if marked as YTD
        if 'Q3' in year_quarters:
            q3 = year_quarters['Q3']

            # Check if Q3 is YTD and requires conversion
            if q3.get('income_statement', {}).get('is_ytd') or q3.get('cash_flow', {}).get('is_ytd'):
                # CRITICAL: Q3 YTD conversion requires BOTH Q1 AND Q2
                missing_quarters = []
                if 'Q1' not in year_quarters:
                    missing_quarters.append('Q1')
                if 'Q2' not in year_quarters:
                    missing_quarters.append('Q2')

                if missing_quarters:
                    fiscal_year = q3.get('fiscal_year', 'Unknown')
                    logger.warning(
                        f"⚠️  YTD CONVERSION SKIPPED: Q3-{fiscal_year} is YTD but {', '.join(missing_quarters)} missing from dataset. "
                        f"Cannot convert YTD to quarterly without prior quarters. This is expected if company didn't file or data is outside range. "
                        f"Marking quarter as conversion_failed."
                    )
                    # Mark as conversion_failed to prevent downstream use
                    q3['ytd_conversion_failed'] = True
                    q3['ytd_conversion_error'] = f"Missing {', '.join(missing_quarters)}"
                    continue  # Skip this Q3 conversion

            # Proceed with conversion only if all required quarters present
            if 'Q2' in year_quarters and 'Q1' in year_quarters:
                q2 = year_quarters['Q2']
                q1 = year_quarters['Q1']

                # Check if income_statement is YTD
                if q3.get('income_statement', {}).get('is_ytd'):
                    income = q3['income_statement']
                    q1_income = q1.get('income_statement', {})
                    q2_income = q2.get('income_statement', {})

                    # Subtract Q1+Q2 from Q3 YTD to get Q3 individual
                    # Note: Q2 may already be converted to individual, so we need Q1+Q2 cumulative
                    for key in income:
                        if key != 'is_ytd' and isinstance(income.get(key), (int, float)):
                            q1_val = q1_income.get(key, 0) or 0
                            q2_val = q2_income.get(key, 0) or 0
                            q3_ytd_val = income[key] or 0

                            # If Q2 was YTD (now converted), use Q1+Q2
                            # Otherwise Q2 is already individual, so Q1+Q2 gives cumulative through Q2
                            q2_cumulative = q1_val + q2_val
                            income[key] = q3_ytd_val - q2_cumulative

                    income['is_ytd'] = False
                    q3_end_date = q3.get('period_end_date', 'Unknown')
                    logger.debug(f"Converted Q3 ending {q3_end_date} income_statement from YTD to individual quarter")

                # Check if cash_flow is YTD
                if q3.get('cash_flow', {}).get('is_ytd'):
                    cash_flow = q3['cash_flow']
                    q1_cash_flow = q1.get('cash_flow', {})
                    q2_cash_flow = q2.get('cash_flow', {})

                    # Subtract Q1+Q2 from Q3 YTD to get Q3 individual
                    for key in cash_flow:
                        if key != 'is_ytd' and isinstance(cash_flow.get(key), (int, float)):
                            q1_val = q1_cash_flow.get(key, 0) or 0
                            q2_val = q2_cash_flow.get(key, 0) or 0
                            q3_ytd_val = cash_flow[key] or 0

                            q2_cumulative = q1_val + q2_val
                            cash_flow[key] = q3_ytd_val - q2_cumulative

                    cash_flow['is_ytd'] = False
                    q3_end_date = q3.get('period_end_date', 'Unknown')
                    logger.debug(f"Converted Q3 ending {q3_end_date} cash_flow from YTD to individual quarter")

    logger.info(f"YTD to quarterly conversion complete for {len(fiscal_year_groups)} fiscal year groups")
    return quarters


def _find_consecutive_quarters(
    periods: List[Dict[str, Any]],
    target_count: int,
    logger
) -> List[Dict[str, Any]]:
    """
    Find the longest sequence of consecutive quarters from a sorted list.

    Consecutive quarters are defined as periods within 60-150 days of each other
    (typical fiscal quarter duration with some tolerance for calendar variations).

    Args:
        periods: List of quarterly periods sorted by period_end_date descending (most recent first)
        target_count: Target number of consecutive quarters (e.g., 4 for TTM)
        logger: Logger instance for warnings

    Returns:
        List of consecutive quarters (may be fewer than target_count if gaps exist)

    Example:
        Input: [Q3-2025 (2025-04-30), Q3-2024 (2024-04-30), Q3-2023 (2023-04-30), Q2-2023 (2023-01-31)]
        Days between: 365 days (NOT consecutive), 365 days (NOT consecutive), 89 days (consecutive)
        Output: [Q3-2023, Q2-2023] - only 2 consecutive quarters
    """
    if not periods:
        return []

    def parse_date(p):
        """Parse period_end_date to datetime, with filed_date as fallback."""
        # Try period_end_date first
        date_str = p.get('period_end_date', '')
        if date_str:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                pass
        # Fallback to filed_date (more reliable for quarter sequencing)
        date_str = p.get('filed_date', '')
        if date_str:
            try:
                if hasattr(date_str, 'isoformat'):
                    return date_str  # Already a datetime
                return datetime.strptime(str(date_str)[:10], '%Y-%m-%d')
            except (ValueError, TypeError):
                pass
        return None

    def is_consecutive_by_fiscal(prev_p, curr_p):
        """Check if two periods are consecutive by fiscal year/period."""
        prev_fy = prev_p.get('fiscal_year')
        curr_fy = curr_p.get('fiscal_year')
        prev_fp = prev_p.get('fiscal_period', '')
        curr_fp = curr_p.get('fiscal_period', '')

        if not all([prev_fy, curr_fy, prev_fp, curr_fp]):
            return False

        # Map periods to quarter numbers
        q_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
        prev_q = q_map.get(prev_fp)
        curr_q = q_map.get(curr_fp)

        if not prev_q or not curr_q:
            return False

        # Consecutive: prev is one quarter after curr (since sorted DESC)
        # e.g., Q3-2025 → Q2-2025 (same year, prev_q=3, curr_q=2)
        # e.g., Q1-2025 → Q4-2024 (year boundary)
        if prev_fy == curr_fy and prev_q == curr_q + 1:
            return True
        if prev_fy == curr_fy + 1 and prev_q == 1 and curr_q == 4:
            return True
        return False

    # Start with the most recent period as the first candidate sequence
    best_sequence = []
    current_sequence = [periods[0]]

    for i in range(1, len(periods)):
        prev_period = periods[i - 1]
        curr_period = periods[i]

        # Primary check: Use fiscal_year + fiscal_period (most reliable)
        is_consecutive = is_consecutive_by_fiscal(prev_period, curr_period)

        # Fallback: Date-based check if fiscal check passes but dates available
        prev_date = parse_date(prev_period)
        curr_date = parse_date(curr_period)
        days_diff = (prev_date - curr_date).days if prev_date and curr_date else None

        if is_consecutive:
            # Consecutive quarter - add to current sequence
            current_sequence.append(curr_period)
            logger.debug(
                f"[CONSECUTIVE_CHECK] ✅ Consecutive (fiscal): "
                f"{prev_period.get('fiscal_period')}-{prev_period.get('fiscal_year')} → "
                f"{curr_period.get('fiscal_period')}-{curr_period.get('fiscal_year')}"
            )

            # If we've reached target, we can return early
            if len(current_sequence) >= target_count:
                logger.info(
                    f"[CONSECUTIVE_CHECK] ✅ Found {len(current_sequence)} consecutive quarters (target: {target_count})"
                )
                return current_sequence[:target_count]
        else:
            # Gap detected - not consecutive
            gap_reason = f"fiscal gap" if days_diff is None else f"{days_diff} days"
            logger.debug(
                f"[CONSECUTIVE_CHECK] ❌ Gap detected: "
                f"{prev_period.get('fiscal_period')}-{prev_period.get('fiscal_year')} → "
                f"{curr_period.get('fiscal_period')}-{curr_period.get('fiscal_year')} "
                f"[{gap_reason}]"
            )

            # Save current sequence if it's the longest so far
            if len(current_sequence) > len(best_sequence):
                best_sequence = current_sequence.copy()

            # Start new sequence from current period
            current_sequence = [curr_period]

    # Check final sequence
    if len(current_sequence) > len(best_sequence):
        best_sequence = current_sequence.copy()

    if len(best_sequence) < target_count:
        best_seq_labels = [f"{p.get('fiscal_period')}-{p.get('fiscal_year')}" for p in best_sequence]
        logger.warning(
            f"[CONSECUTIVE_CHECK] ⚠️  Could not find {target_count} consecutive quarters. "
            f"Best sequence: {len(best_sequence)} quarters - {best_seq_labels}"
        )

    return best_sequence[:target_count] if len(best_sequence) >= target_count else best_sequence


def get_rolling_ttm_periods(
    all_periods: List[Dict[str, Any]],
    compute_missing: bool = True,
    num_quarters: int = 4
) -> List[Dict[str, Any]]:
    """
    Get the most recent N quarters for TTM or historical analysis.

    ═══════════════════════════════════════════════════════════════════════════
    CRITICAL: FY Period Filtering Strategy
    ═══════════════════════════════════════════════════════════════════════════

    **INPUT EXPECTATION:**
    - all_periods MAY contain both FY and Q periods (e.g., from database query)
    - FY periods are needed as input for Q4 computation

    **PROCESSING:**
    1. Separate FY periods from quarterly periods
    2. Convert YTD values to individual quarters (Q2/Q3 normalization)
    3. Compute Q4 from FY data: Q4 = FY - (Q1 + Q2 + Q3)
       - New Q4 periods are labeled fiscal_period='Q4' with is_ytd=False
    4. Add computed Q4 periods to quarterly_periods list

    **OUTPUT GUARANTEE:**
    - Returns ONLY Q periods (Q1, Q2, Q3, Q4)
    - NO FY periods in returned list
    - FY periods are consumed during Q4 computation, then discarded
    - All returned quarters are Point-In-Time (is_ytd=False after conversion)

    **WHY FY PERIODS ARE NOT RETURNED:**
    - FY periods contain YTD cumulative totals
    - Q4 periods contain PIT individual quarter values
    - Including both would double-count Q4 data in TTM calculations
    - Downstream consumers (DCF, TTM analysis) expect only Q periods

    ═══════════════════════════════════════════════════════════════════════════

    Strategy:
    1. Sort periods by fiscal_year DESC, then fiscal_period DESC (most recent first)
    2. Separate FY and quarterly periods
    3. Convert YTD cumulative values to individual quarters (Q2/Q3)
    4. If we have recent Q1-Q3 + FY from same fiscal year → compute Q4
       - Match by period_end_date proximity (FY within ~90 days of Q3)
    5. Return N most recent quarters (may span 2+ fiscal years)
    6. **FY periods are NOT included in output**

    Args:
        all_periods: List of period dicts with period_end_date, fiscal_period (may include FY)
        compute_missing: Whether to compute Q4 from FY if available
        num_quarters: Number of quarters to return (default 4 for TTM, use 12 for geometric mean)

    Returns:
        List of N most recent quarterly periods (Q1-Q4 only, NO FY periods)
        All values are Point-In-Time (YTD converted, Q4 computed)

    Use Cases:
        - num_quarters=4: TTM (Trailing Twelve Months) for simple valuation (DEFAULT)
        - num_quarters=12: 3-year trend for geometric mean, stable growth rates

    Example:
        >>> periods = [
        ...     {'period_end_date': '2025-06-27', 'fiscal_period': 'FY', 'ocf': 110543},
        ...     {'period_end_date': '2025-03-28', 'fiscal_period': 'Q3', 'ocf': 91443},
        ...     {'period_end_date': '2024-12-27', 'fiscal_period': 'Q2', 'ocf': 62585},
        ...     {'period_end_date': '2024-09-27', 'fiscal_period': 'Q1', 'ocf': 39895},
        ... ]
        >>> ttm_only = get_rolling_ttm_periods(periods, num_quarters=4)
        >>> # Returns: [Q4-2025 (computed), Q3-2025, Q2-2025, Q1-2025]
        >>> # FY-2025 is NOT in the output (consumed for Q4 computation)
    """
    if not all_periods:
        return []

    # Normalize and validate fiscal_year/fiscal_period using frame if available
    def normalize_period_data(period: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and correct fiscal_year/fiscal_period using frame key.

        SEC frame format: "CY2025Q3" or "FY2025Q1" (Calendar/Fiscal Year + Quarter)
        Frame is more reliable than separate fy/fp fields which can have manual entry errors.

        Args:
            period: Period dict with frame, fiscal_year, fiscal_period keys

        Returns:
            Period dict with corrected fiscal_year/fiscal_period
        """
        frame = period.get('frame', '')
        fy = period.get('fiscal_year')
        fp = period.get('fiscal_period', '')

        if frame:
            # Extract year and quarter from frame (e.g., "CY2025Q3" -> year=2025, period="Q3")
            try:
                # Frame format: [C|F]Y followed by year and optionally Q[1-4]
                if 'Q' in frame:
                    # Quarterly frame: "CY2025Q3" or "FY2025Q1"
                    parts = frame.split('Q')
                    year_part = parts[0]  # "CY2025" or "FY2025"
                    quarter_num = parts[1]  # "3"

                    # Extract year (last 4 digits of year_part)
                    frame_year = int(year_part[-4:])
                    frame_period = f"Q{quarter_num}"
                else:
                    # Annual frame: "CY2025" or "FY2025"
                    frame_year = int(frame[-4:])
                    frame_period = "FY"

                # Validate against existing fy/fp
                if fy and fy != frame_year:
                    logger.debug(
                        f"Fiscal year mismatch: frame={frame} indicates {frame_year}, "
                        f"but fiscal_year={fy}. Using frame value {frame_year}."
                    )
                if fp and fp != frame_period:
                    logger.debug(
                        f"Fiscal period mismatch: frame={frame} indicates {frame_period}, "
                        f"but fiscal_period={fp}. Using frame value {frame_period}."
                    )

                # Correct using frame (authoritative source)
                period['fiscal_year'] = frame_year
                period['fiscal_period'] = frame_period

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse frame '{frame}': {e}. Using existing fy/fp.")

        return period

    # Normalize all periods using frame validation
    all_periods = [normalize_period_data(p) for p in all_periods]

    # DEBUG: Check fiscal periods AFTER normalization
    fiscal_periods_after_norm = [p.get('fiscal_period') for p in all_periods]
    logger.info(f"[QUARTERLY_CALC_DEBUG] Fiscal periods AFTER normalization: {fiscal_periods_after_norm}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Separate FY and quarterly periods
    # FY periods will be used for Q4 computation, then DISCARDED from output
    # ═══════════════════════════════════════════════════════════════════════════
    fy_periods = [p for p in all_periods if p.get('fiscal_period') == 'FY']
    quarterly_periods = [p for p in all_periods if p.get('fiscal_period', '').startswith('Q')]

    logger.info(f"[QUARTERLY_CALC_DEBUG] After separation: {len(fy_periods)} FY periods, {len(quarterly_periods)} Q periods")

    # Helper function for date parsing (used for YTD conversion and Q4 proximity matching)
    def parse_date(p):
        """Parse period_end_date for date-based proximity matching."""
        date_str = p.get('period_end_date', '')
        if not date_str:
            return datetime.min
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            logger.warning(f"Invalid period_end_date format: {date_str}")
            return datetime.min

    # Sort by fiscal_year and fiscal_period (most recent first)
    # This is the natural ordering for quarterly financial data
    # Use FiscalPeriodService for consistent period sorting
    fiscal_period_service = get_fiscal_period_service()

    # Sort DESC by fiscal_year, then DESC by fiscal_period (2025-Q4 > 2025-Q3 > ... > 2024-Q4)
    quarterly_periods.sort(key=lambda p: (p.get('fiscal_year', 0), fiscal_period_service.get_period_sort_key(p.get('fiscal_period', 'Q1'))), reverse=True)
    fy_periods.sort(key=lambda p: (p.get('fiscal_year', 0), fiscal_period_service.get_period_sort_key(p.get('fiscal_period', 'FY'))), reverse=True)

    # CRITICAL: Convert YTD to individual quarters BEFORE any calculations
    quarterly_periods = convert_ytd_to_quarterly(quarterly_periods)

    # ═══════════════════════════════════════════════════════════════════════════
    # Compute missing Q4 for recent fiscal years if needed
    # REFACTORED: Use chronological grouping instead of fiscal_year labels
    # OPTIMIZATION: Only compute Q4s until we have enough quarters for TTM calculations
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info(f"[Q4_COMPUTE] Before condition: compute_missing={compute_missing}, quarterly_periods={len(quarterly_periods)}, fy_periods={len(fy_periods)}")

    if compute_missing and len(quarterly_periods) >= 3 and len(fy_periods) >= 1:
        logger.info(f"[Q4_COMPUTE] ✅ Entering Q4 computation block: {len(fy_periods)} FY periods, {len(quarterly_periods)} Q periods")

        # OPTIMIZATION: Define minimum quarters needed for TTM calculations
        # This is configurable - if you need 16 quarters, change this value and the logic adapts
        min_quarters_for_ttm = num_quarters  # Use requested quarters (default 4, could be 8, 12, 16)
        current_quarterly_count = len(quarterly_periods)

        # Initialize computed_q4s before conditional logic to avoid UnboundLocalError
        computed_q4s = []

        logger.info(f"[Q4_COMPUTE] Current quarters: {current_quarterly_count}, Target: {min_quarters_for_ttm}")

        # CRITICAL FIX: Always attempt Q4 computation for ALL FY periods
        # Don't skip based on current count - YTD filtering happens later and may reduce count
        logger.info(f"[Q4_COMPUTE] Will attempt to compute Q4 for ALL {len(fy_periods)} FY periods where quarterly data available")

        # Process each FY period chronologically
        # CRITICAL FIX: Compute Q4 for ALL FY periods, not just until target reached
        # Rationale: YTD filtering may later reduce quarter count, creating gaps
        # This ensures we have complete Q4 data to eliminate 184-day gaps between Q1 and Q3
        for fy_idx, fy in enumerate(fy_periods):
                logger.info(f"[Q4_COMPUTE] Processing FY {fy_idx+1}/{len(fy_periods)}: fiscal_year={fy.get('fiscal_year')}, period_end={fy.get('period_end_date')}")
                fy_end_date = parse_date(fy)
                if fy_end_date == datetime.min:
                    logger.info(f"[Q4_COMPUTE] ❌ FY has invalid date, skipping")
                    continue

                # Find Q1, Q2, Q3 that belong to this fiscal year
                # Strategy: Quarters whose period_end_date is within ~365 days before FY end date
                # and closest to FY end date (latest Q3 should be within ~90 days of FY)

                q1, q2, q3 = None, None, None

                # Find Q3 first (should be ~90 days before FY end)
                logger.info(f"[Q4_COMPUTE] Searching for Q3 matching FY ending {fy.get('period_end_date')}")
                q3_candidates = []
                for q in quarterly_periods:
                    if q.get('fiscal_period') == 'Q3':
                        q_end_date = parse_date(q)
                        days_diff = (fy_end_date - q_end_date).days
                        q3_candidates.append((q.get('fiscal_year'), q.get('period_end_date'), days_diff))
                        # Q3 should end 60-120 days before FY (typical fiscal quarter)
                        if 30 <= days_diff <= 150:
                            q3 = q
                            logger.info(f"[Q4_COMPUTE] ✅ Found Q3: fiscal_year={q.get('fiscal_year')}, period_end={q.get('period_end_date')}, days_before_FY={days_diff}")
                            break

                if not q3 and q3_candidates:
                    logger.info(f"[Q4_COMPUTE] Q3 candidates found but none matched (30-150 days): {q3_candidates}")

                # FALLBACK 1: If no Q3 found, try relaxed proximity (30-180 days)
                if not q3:
                    logger.warning(
                        f"[Q4_COMPUTE] No Q3 found within 30-150 days for FY {fy.get('fiscal_year')} ending {fy.get('period_end_date')}, "
                        f"trying relaxed proximity (30-180 days)"
                    )
                    for q in quarterly_periods:
                        if q.get('fiscal_period') == 'Q3':
                            q_end_date = parse_date(q)
                            days_diff = (fy_end_date - q_end_date).days
                            if 30 <= days_diff <= 180:
                                q3 = q
                                logger.info(
                                    f"[Q4_COMPUTE] ✅ Found Q3 with relaxed proximity: fiscal_year={q.get('fiscal_year')}, "
                                    f"period_end={q.get('period_end_date')}, days_before_FY={days_diff}"
                                )
                                break

                # FALLBACK 2: If still no Q3 found, try fiscal year match only (ignore proximity)
                if not q3:
                    logger.warning(
                        f"[Q4_COMPUTE] No Q3 found within 30-180 days, attempting fiscal year match only"
                    )
                    for q in quarterly_periods:
                        if q.get('fiscal_period') == 'Q3' and q.get('fiscal_year') == fy.get('fiscal_year'):
                            q3 = q
                            logger.info(
                                f"[Q4_COMPUTE] ✅ Found Q3 by fiscal year match (ignoring proximity): "
                                f"fiscal_year={q.get('fiscal_year')}, period_end={q.get('period_end_date')}"
                            )
                            break

                # If no Q3 found, can't compute Q4 reliably
                if not q3:
                    logger.warning(f"[Q4_COMPUTE] ❌ No Q3 found for FY {fy.get('fiscal_year')} ending {fy.get('period_end_date')}, skipping Q4 computation")
                    continue

                q3_end_date = parse_date(q3)
                fy_year = fy.get('fiscal_year')

                # Find Q2 - first try date proximity, then fiscal year match
                for q in quarterly_periods:
                    if q.get('fiscal_period') == 'Q2':
                        q_end_date = parse_date(q)
                        if q_end_date and q3_end_date:
                            days_diff = (q3_end_date - q_end_date).days
                            if 30 <= days_diff <= 150:
                                q2 = q
                                break
                # Fallback: fiscal year match
                if not q2:
                    for q in quarterly_periods:
                        if q.get('fiscal_period') == 'Q2' and q.get('fiscal_year') == fy_year:
                            q2 = q
                            break

                # Find Q1 - first try date proximity, then fiscal year match
                for q in quarterly_periods:
                    if q.get('fiscal_period') == 'Q1':
                        q_end_date = parse_date(q)
                        if q2:
                            q2_end_date = parse_date(q2)
                            if q_end_date and q2_end_date:
                                days_diff = (q2_end_date - q_end_date).days
                                if 30 <= days_diff <= 150:
                                    q1 = q
                                    break
                        elif q3_end_date and q_end_date:
                            # No Q2, try matching to Q3
                            days_diff = (q3_end_date - q_end_date).days
                            if 120 <= days_diff <= 250:
                                q1 = q
                                break
                # Fallback: fiscal year match
                if not q1:
                    for q in quarterly_periods:
                        if q.get('fiscal_period') == 'Q1' and q.get('fiscal_year') == fy_year:
                            q1 = q
                            break

                # Check if Q4 already exists for this fiscal year
                # (within ~30 days of FY end date)
                has_q4 = False
                for q in quarterly_periods:
                    if q.get('fiscal_period') == 'Q4':
                        q_end_date = parse_date(q)
                        days_diff = abs((fy_end_date - q_end_date).days)
                        if days_diff <= 30:
                            has_q4 = True
                            break

                if has_q4:
                    logger.debug(f"Q4 already exists for FY ending {fy.get('period_end_date')}, skipping computation")
                    continue

                # If we have FY + at least 2 quarters, compute Q4
                available_quarters = [q for q in [q1, q2, q3] if q is not None]
                logger.info(f"[Q4_COMPUTE] Available quarters for computation: Q1={q1 is not None}, Q2={q2 is not None}, Q3={q3 is not None} (count={len(available_quarters)})")

                if len(available_quarters) >= 2:
                    logger.info(f"[Q4_COMPUTE] 🔄 Computing Q4 from FY={fy.get('fiscal_year')}")
                    q4_computed = compute_missing_quarter(fy, q1, q2, q3)

                    if q4_computed:
                        # Set period_end_date for computed Q4 (same as FY)
                        q4_computed['period_end_date'] = fy.get('period_end_date')

                        fiscal_year_display = fy.get('fiscal_year', 'Unknown')
                        logger.info(
                            f"✅ Computed Q4 for FY ending {fy.get('period_end_date')} (fiscal_year={fiscal_year_display}). "
                            f"OCF: {q4_computed.get('cash_flow', {}).get('operating_cash_flow', 0)/1e6:.1f}M, "
                            f"CapEx: {abs(q4_computed.get('cash_flow', {}).get('capital_expenditures', 0))/1e6:.1f}M"
                        )
                        computed_q4s.append(q4_computed)
                    else:
                        logger.info(f"[Q4_COMPUTE] ❌ compute_missing_quarter() returned None for FY={fy.get('fiscal_year')}")
                else:
                    logger.info(f"[Q4_COMPUTE] ❌ Not enough quarters available (need >= 2, got {len(available_quarters)}) for FY={fy.get('fiscal_year')}")

        # Add computed Q4s to quarterly periods and re-sort by fiscal_year + fiscal_period
        quarterly_periods.extend(computed_q4s)
        fiscal_service = get_fiscal_period_service()
        quarterly_periods.sort(key=lambda p: (p.get('fiscal_year', 0), fiscal_service.get_period_sort_key(p.get('fiscal_period', ''))), reverse=True)

        logger.info(f"[Q4_COMPUTE] ✅ Computed {len(computed_q4s)} Q4 periods. Total quarterly periods after Q4 computation: {len(quarterly_periods)}")
    else:
        logger.info(f"[Q4_COMPUTE] ❌ Skipping Q4 computation block (condition not met)")

    # Filter out only periods where YTD conversion FAILED
    # CRITICAL FIX: is_ytd=True means original source was YTD (should be converted by now)
    # ytd_conversion_failed=True means conversion failed and data is still YTD (must skip)
    non_ytd_periods = []
    for period in quarterly_periods:
        income_failed = period.get('income_statement', {}).get('ytd_conversion_failed', False)
        cash_flow_failed = period.get('cash_flow', {}).get('ytd_conversion_failed', False)

        if income_failed or cash_flow_failed:
            logger.warning(
                f"⚠️  Skipping {period.get('fiscal_year')}-{period.get('fiscal_period')} "
                f"(income_failed={income_failed}, cash_flow_failed={cash_flow_failed}) - "
                f"YTD data couldn't be converted due to missing prior quarters"
            )
        else:
            non_ytd_periods.append(period)

    # CRITICAL FIX: Sort by actual calendar date (period_end_date) for chronological order
    # This ensures TTM periods are truly consecutive, not just by fiscal_year + fiscal_period
    non_ytd_periods.sort(key=lambda p: parse_date(p), reverse=True)

    period_labels = [f"{p.get('fiscal_period')}-{p.get('fiscal_year')} ({p.get('period_end_date')})" for p in non_ytd_periods[:8]]
    logger.info(
        f"[TTM_SELECT] After YTD filter and date sort: {len(non_ytd_periods)} periods available. "
        f"Periods: {period_labels}"
    )

    # Find longest consecutive sequence of quarters for valid TTM calculation
    # TTM requires consecutive quarters (within 60-120 days of each other)
    consecutive_periods = _find_consecutive_quarters(non_ytd_periods, num_quarters, logger)

    if len(consecutive_periods) < num_quarters:
        logger.warning(
            f"⚠️  Only {len(consecutive_periods)} consecutive quarters available (requested {num_quarters}). "
            f"TTM calculation may be invalid with non-consecutive periods!"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL OUTPUT: Returns ONLY Q periods (Q1-Q4), NO FY periods
    # FY periods were consumed for Q4 computation and are not included
    # ═══════════════════════════════════════════════════════════════════════════
    return consecutive_periods


def analyze_quarterly_patterns(
    quarters: List[Dict[str, Any]],
    metric_key: str = 'operating_cash_flow'
) -> Dict[str, Any]:
    """
    Analyze quarterly patterns using TTM-based YoY comparison (fiscal-year agnostic).

    Computes:
    - TTM YoY growth: Current TTM (Q0-Q3) vs Prior TTM (Q4-Q7)
    - Sequential growth: Quarter-over-quarter changes
    - Seasonality variance: Coefficient of variation across recent quarters
    - Overall trend: Improving/declining/stable

    Args:
        quarters: List of 8+ quarters sorted most recent first (MUST exclude FY periods)
        metric_key: Metric to analyze (default: operating_cash_flow)

    Returns:
        Dict with analysis results including:
        - avg_yoy_growth: TTM-based year-over-year growth rate
        - avg_sequential_growth: Average quarter-over-quarter growth
        - seasonality_variance: Variance across quarters (%)
        - trend: 'accelerating', 'decelerating', 'stable'

    Example:
        >>> quarters = get_rolling_ttm_periods(all_periods, num_quarters=8)
        >>> patterns = analyze_quarterly_patterns(quarters, 'free_cash_flow')
        >>> print(f"TTM YoY growth: {patterns['avg_yoy_growth']:.1f}%")
    """
    if len(quarters) < 8:
        logger.warning(f"Need 8 quarters for TTM pattern analysis, got {len(quarters)}")
        return {}

    # NOTE: Do NOT filter out FY periods here - get_rolling_ttm_periods() already handles
    # FY → Q4 conversion and returns only quarterly periods (Q1-Q4). If FY periods remain,
    # they contain YTD-normalized Q4 data and should be included.

    # Use quarters as-is (already processed by get_rolling_ttm_periods)
    quarterly_only = quarters

    # Need 12 quarters for geometric mean approach (3 years of TTM periods)
    if len(quarterly_only) < 12:
        logger.warning(
            f"Only {len(quarterly_only)} quarters available (need 12 for geometric mean). "
            f"Using simple TTM growth to avoid inflated rates from incomplete prior2 period."
        )
        # Fall back to 8-quarter simple growth if we have at least 8
        if len(quarterly_only) >= 8:
            logger.info("Falling back to simple 8-quarter TTM growth calculation")
        else:
            return {}

    # Extract values for the metric
    values = []
    num_quarters = min(12, len(quarterly_only))  # Use up to 12 quarters
    for q in quarterly_only[:num_quarters]:
        val = extract_nested_value(q, metric_key)
        values.append(val or 0)

    # === TTM-BASED YOY GROWTH WITH GEOMETRIC MEAN (Fiscal-year agnostic) ===
    # Calculate 3 TTM periods for geometric mean
    # Current TTM (Q0-Q3): Most recent 4 quarters
    # Prior1 TTM (Q4-Q7): One year back
    # Prior2 TTM (Q8-Q11): Two years back (if available)

    current_ttm = sum(values[0:4])
    prior1_ttm = sum(values[4:8]) if len(values) >= 8 else 0
    prior2_ttm = sum(values[8:12]) if len(values) >= 12 else 0

    # Calculate growth rates
    growth_current_vs_prior1 = ((current_ttm / prior1_ttm) - 1) * 100 if prior1_ttm > 0 else 0
    growth_prior1_vs_prior2 = ((prior1_ttm / prior2_ttm) - 1) * 100 if prior2_ttm > 0 else 0

    # CAGR (Compound Annual Growth Rate) over 2 years: Direct endpoint calculation
    # EDGE CASE: Only use CAGR if we have EXACTLY 12 quarters
    # If we have 10-11 quarters, prior2_ttm will be incomplete (missing 1-2 quarters)
    # which can inflate growth rates due to smaller base. Use simple growth instead.
    used_geometric_mean = False
    avg_yoy_growth = 0  # Initialize to ensure it's always defined

    if len(values) == 12 and prior2_ttm > 0:
        # SIMPLIFIED VALIDATION: Calculate CAGR directly from endpoints (current to prior2)
        # This approach is more robust than geometric mean of intermediate growth rates:
        # - Handles negative intermediate values (prior1) gracefully
        # - Only requires that endpoint ratio is positive (i.e., not both negative)
        # - True mathematical CAGR: (final/initial)^(1/years) - 1

        # Calculate overall growth ratio from prior2 to current (2-year period)
        overall_ratio = current_ttm / prior2_ttm

        # VALIDATION: Check if ratio is positive (growth > -100%)
        # If ratio <= 0, it means both values have same sign and absolute decline >= 100%
        # Example: 20 / -10 = -2.0 (invalid), -20 / 10 = -2.0 (invalid), -20 / -10 = 2.0 (valid)
        if overall_ratio <= 0:
            logger.warning(
                f"⚠️ Overall growth ratio is non-positive (Current: ${current_ttm/1e9:.2f}B, "
                f"Prior2: ${prior2_ttm/1e9:.2f}B, Ratio: {overall_ratio:.4f}). "
                f"This indicates sign change with >100% absolute decline. "
                f"CAGR calculation invalid. Using simple growth instead."
            )
            # Fall through to simple growth calculation
        else:
            # Safe to calculate 2-year CAGR
            # Formula: (final/initial)^(1/2) - 1 for 2-year period
            cagr = (overall_ratio ** 0.5) - 1
            avg_yoy_growth = cagr * 100
            used_geometric_mean = True  # Keep flag name for backward compatibility

            logger.info(
                f"🔍 TTM YoY Growth (2-Year CAGR - 12 quarters):\n"
                f"  Current TTM: ${current_ttm/1e9:.2f}B\n"
                f"  Prior1 TTM:  ${prior1_ttm/1e9:.2f}B (not used in CAGR)\n"
                f"  Prior2 TTM:  ${prior2_ttm/1e9:.2f}B (2 years ago)\n"
                f"  Overall Ratio: {overall_ratio:.4f}x\n"
                f"  2-Year CAGR: {avg_yoy_growth:+.1f}%"
            )
    elif len(values) >= 8 and prior1_ttm > 0:
        # Simple growth if we don't have exactly 12 quarters OR if data quality issues prevent geometric mean
        # This prevents issues from:
        # - 10-11 quarters (incomplete prior2_ttm)
        # - Zero/negative TTM values (division by zero, invalid growth rates)
        avg_yoy_growth = growth_current_vs_prior1
        logger.info(
            f"🔍 TTM YoY Growth (Simple - {len(values)} quarters):\n"
            f"  Current TTM: ${current_ttm/1e9:.2f}B\n"
            f"  Prior1 TTM:  ${prior1_ttm/1e9:.2f}B\n"
            f"  Growth: {avg_yoy_growth:.1f}%"
        )
        # Only warn if we're close to 12 quarters but not quite there (10-11)
        # If we have exactly 12, it means we fell through due to data quality (zero/negative values)
        if 10 <= len(values) < 12:
            logger.warning(
                f"⚠️ Only {len(values)} quarters available (need 12 for geometric mean). "
                f"Using simple TTM growth to avoid inflated rates from incomplete prior2 period."
            )
        elif len(values) == 12:
            # We have 12 quarters but geometric mean wasn't used due to data quality
            logger.debug(
                f"ℹ️ Have 12 quarters but using simple growth due to data quality: "
                f"current_ttm=${current_ttm/1e9:.2f}B, prior1_ttm=${prior1_ttm/1e9:.2f}B, "
                f"prior2_ttm=${prior2_ttm/1e9:.2f}B (geometric mean requires all >0)"
            )
    else:
        avg_yoy_growth = 0
        logger.warning("Insufficient data for TTM YoY growth calculation")

    # === SEQUENTIAL GROWTH (QoQ) ===
    sequential_growth = []
    for i in range(min(4, len(values) - 1)):
        current = values[i]
        prev_q = values[i + 1]

        if prev_q > 0:
            growth_pct = ((current - prev_q) / prev_q) * 100
            sequential_growth.append(growth_pct)

    avg_sequential_growth = sum(sequential_growth) / len(sequential_growth) if sequential_growth else 0

    # === SEASONALITY VARIANCE ===
    # Calculate coefficient of variation for most recent 4 quarters
    recent_4q = values[0:4]
    mean_val = sum(recent_4q) / 4 if recent_4q else 0
    if mean_val > 0:
        variance = sum((v - mean_val) ** 2 for v in recent_4q) / 4
        std_dev = variance ** 0.5
        seasonality_variance = (std_dev / mean_val * 100)
    else:
        seasonality_variance = 0

    # === TREND DETERMINATION ===
    # DEFENSIVE: Ensure avg_yoy_growth is a real number (not complex) before comparison
    # This should never happen after the pre-validation above, but adding as safety check
    if isinstance(avg_yoy_growth, complex):
        logger.error(
            f"❌ CRITICAL: avg_yoy_growth is complex number ({avg_yoy_growth}). "
            f"This indicates a bug in geometric mean calculation. Forcing to 0."
        )
        avg_yoy_growth = 0  # Force to real number

    if avg_yoy_growth > 10:
        trend = 'accelerating'
    elif avg_yoy_growth < -5:
        trend = 'decelerating'
    else:
        trend = 'stable'

    seasonality_classification = "high" if seasonality_variance >= 25 else "moderate" if seasonality_variance >= 10 else "low"
    yoy_growth_payload = {
        "ttm_pct": avg_yoy_growth,
        "method": "cagr" if used_geometric_mean else "simple",
        "current_ttm": current_ttm,
        "prior_ttm": prior1_ttm,
    }

    return {
        'metric': metric_key,
        'avg_yoy_growth': avg_yoy_growth,
        'avg_sequential_growth': avg_sequential_growth,
        'seasonality_variance': seasonality_variance,
        'seasonality': {
            'classification': seasonality_classification,
            'variance_pct': seasonality_variance,
        },
        'yoy_growth': yoy_growth_payload,
        'trend': trend,
        'current_ttm': current_ttm,
        'prior_ttm': prior1_ttm,  # FIX: Use prior1_ttm (the actual variable name)
        'quarters_analyzed': len(quarterly_only[:12]),  # Report actual quarters analyzed (up to 12 for geometric mean)
        'used_geometric_mean': used_geometric_mean  # Flag to skip sector caps if True
    }


@dataclass
class Q4ComputationResult:
    """Result of Q4 fallback calculation with quality metadata."""
    q4_data: Optional[Dict[str, Any]]
    method: str  # 'exact', 'proportional', 'annual_average', 'none'
    confidence: float  # 0.0-1.0
    warnings: List[str]
    metrics_computed: List[str]

    def is_valid(self) -> bool:
        """Check if Q4 was successfully computed."""
        return self.q4_data is not None and self.method != 'none'


def calculate_q4_with_fallback(
    fy_data: Dict[str, Any],
    q1_data: Optional[Dict[str, Any]] = None,
    q2_data: Optional[Dict[str, Any]] = None,
    q3_data: Optional[Dict[str, Any]] = None
) -> Q4ComputationResult:
    """
    Calculate Q4 with multi-strategy fallback chain.

    Strategy hierarchy:
    1. Exact: Q4 = FY - (Q1 + Q2 + Q3) when all quarters available
    2. Proportional: Estimate from available quarters' patterns
    3. Annual Average: Q4 = FY / 4 when no quarterly data

    Args:
        fy_data: Full year data dict (fiscal_period='FY')
        q1_data: Q1 data dict (optional)
        q2_data: Q2 data dict (optional)
        q3_data: Q3 data dict (optional)

    Returns:
        Q4ComputationResult with computed Q4 data, method used, and confidence

    Example:
        >>> result = calculate_q4_with_fallback(fy, q1, q2, q3)
        >>> if result.is_valid():
        ...     print(f"Q4 computed via {result.method} (confidence: {result.confidence:.0%})")
    """
    warnings = []
    available_quarters = [q for q in [q1_data, q2_data, q3_data] if q is not None]

    if not fy_data:
        return Q4ComputationResult(
            q4_data=None,
            method='none',
            confidence=0.0,
            warnings=["No FY data provided"],
            metrics_computed=[]
        )

    fiscal_year = fy_data.get('fiscal_year')

    # Strategy 1: Exact calculation (highest confidence)
    if len(available_quarters) >= 2:
        q4_computed = compute_missing_quarter(fy_data, q1_data, q2_data, q3_data)
        if q4_computed:
            # Calculate confidence based on available quarters
            confidence = 0.95 if len(available_quarters) == 3 else 0.85

            # Check for data quality issues
            cash_flow = q4_computed.get('cash_flow', {})
            income = q4_computed.get('income_statement', {})

            # Validate: negative revenue or unusually large values reduce confidence
            if cash_flow.get('operating_cash_flow', 0) < 0:
                warnings.append("Computed OCF is negative")
                confidence *= 0.8
            if income.get('total_revenue', 0) < 0:
                warnings.append("Computed revenue is negative")
                confidence *= 0.7

            metrics_computed = [k for k in cash_flow.keys() if k != 'is_ytd']
            metrics_computed += [k for k in income.keys() if k != 'is_ytd']

            logger.info(
                f"✅ Q4 computed via EXACT method for FY {fiscal_year} "
                f"(confidence: {confidence:.0%}, {len(available_quarters)} quarters used)"
            )

            return Q4ComputationResult(
                q4_data=q4_computed,
                method='exact',
                confidence=confidence,
                warnings=warnings,
                metrics_computed=metrics_computed
            )

    # Strategy 2: Proportional estimation
    # Use available quarters to estimate Q4 proportionally
    if len(available_quarters) == 1:
        q_available = available_quarters[0]
        q_period = q_available.get('fiscal_period', '')

        # Estimate based on single quarter pattern
        # Q4 is typically 25-30% of annual (some seasonality)
        q4_computed = _estimate_q4_proportional(fy_data, available_quarters)

        if q4_computed:
            # Lower confidence for proportional estimation
            confidence = 0.65
            warnings.append(f"Q4 estimated proportionally from {q_period} only")

            logger.info(
                f"🔄 Q4 computed via PROPORTIONAL method for FY {fiscal_year} "
                f"(confidence: {confidence:.0%}, using {q_period})"
            )

            return Q4ComputationResult(
                q4_data=q4_computed,
                method='proportional',
                confidence=confidence,
                warnings=warnings,
                metrics_computed=list(q4_computed.get('cash_flow', {}).keys())
            )

    # Strategy 3: Annual average (lowest confidence)
    # Simple FY / 4 when no quarterly data
    if len(available_quarters) == 0:
        q4_computed = _estimate_q4_annual_average(fy_data)

        if q4_computed:
            confidence = 0.50
            warnings.append("Q4 estimated as FY/4 (no quarterly data available)")

            logger.warning(
                f"⚠️ Q4 computed via ANNUAL_AVERAGE method for FY {fiscal_year} "
                f"(confidence: {confidence:.0%}, no quarterly data)"
            )

            return Q4ComputationResult(
                q4_data=q4_computed,
                method='annual_average',
                confidence=confidence,
                warnings=warnings,
                metrics_computed=list(q4_computed.get('cash_flow', {}).keys())
            )

    # No valid computation possible
    return Q4ComputationResult(
        q4_data=None,
        method='none',
        confidence=0.0,
        warnings=["Could not compute Q4 with available data"],
        metrics_computed=[]
    )


def _estimate_q4_proportional(
    fy_data: Dict[str, Any],
    available_quarters: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Estimate Q4 proportionally from available quarters.

    Uses the ratio of available quarters to estimate Q4's share.
    """
    if not available_quarters:
        return None

    fiscal_year = fy_data.get('fiscal_year')

    # Calculate sum of available quarters
    quarter_sum = {}
    for q in available_quarters:
        cf = q.get('cash_flow', {})
        for key, val in cf.items():
            if key != 'is_ytd' and isinstance(val, (int, float)):
                quarter_sum[key] = quarter_sum.get(key, 0) + val

        income = q.get('income_statement', {})
        for key, val in income.items():
            if key != 'is_ytd' and isinstance(val, (int, float)):
                quarter_sum[key] = quarter_sum.get(key, 0) + val

    # Estimate Q4 = FY - sum of available quarters
    # But scale by expected proportion (if 1 quarter, assume it's ~25% of FY)
    num_quarters = len(available_quarters)
    expected_proportion = num_quarters / 4.0

    q4_computed = {
        'symbol': fy_data.get('symbol'),
        'fiscal_year': fiscal_year,
        'fiscal_period': 'Q4',
        'computed': True,
        'computation_method': 'proportional_estimate',
        'cash_flow': {'is_ytd': False},
        'income_statement': {'is_ytd': False},
    }

    # For each metric in FY, estimate Q4
    fy_cf = fy_data.get('cash_flow', {})
    for key, fy_val in fy_cf.items():
        if key != 'is_ytd' and isinstance(fy_val, (int, float)):
            q_sum = quarter_sum.get(key, 0)
            # Estimate remaining value for Q4
            # If q_sum represents expected_proportion of FY, scale accordingly
            if q_sum != 0:
                remaining = fy_val - (q_sum / expected_proportion * expected_proportion)
                q4_computed['cash_flow'][key] = remaining / (1 - expected_proportion) if expected_proportion < 1 else fy_val / 4
            else:
                q4_computed['cash_flow'][key] = fy_val / 4

    return q4_computed


def _estimate_q4_annual_average(fy_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Estimate Q4 as FY / 4 (simple annual average).

    Lowest confidence method, used when no quarterly data available.
    """
    fiscal_year = fy_data.get('fiscal_year')

    q4_computed = {
        'symbol': fy_data.get('symbol'),
        'fiscal_year': fiscal_year,
        'fiscal_period': 'Q4',
        'computed': True,
        'computation_method': 'annual_average',
        'cash_flow': {'is_ytd': False},
        'income_statement': {'is_ytd': False},
    }

    # Simply divide FY by 4
    fy_cf = fy_data.get('cash_flow', {})
    for key, fy_val in fy_cf.items():
        if key != 'is_ytd' and isinstance(fy_val, (int, float)):
            q4_computed['cash_flow'][key] = fy_val / 4

    fy_income = fy_data.get('income_statement', {})
    for key, fy_val in fy_income.items():
        if key != 'is_ytd' and isinstance(fy_val, (int, float)):
            q4_computed['income_statement'][key] = fy_val / 4

    return q4_computed


@dataclass
class TTMResult:
    """Result of weighted TTM calculation with quality metadata."""
    ttm_data: Dict[str, float]
    quarters_used: int
    quality_score: float  # 0-100
    scaling_applied: bool
    warnings: List[str]

    def is_complete(self) -> bool:
        """Check if TTM is based on complete 4 quarters."""
        return self.quarters_used >= 4 and not self.scaling_applied


def calculate_ttm_weighted(
    quarters: List[Dict[str, Any]],
    weights: Optional[List[float]] = None
) -> TTMResult:
    """
    Calculate TTM with weighted quarters and quality scoring.

    Supports partial TTM calculation when fewer than 4 quarters available,
    scaling up to 4-quarter equivalent with reduced confidence.

    Args:
        quarters: List of quarterly periods (most recent first)
        weights: Optional weights for each quarter (default: equal weights)
                 More recent quarters can be weighted higher

    Returns:
        TTMResult with:
        - ttm_data: Dict of TTM metrics
        - quarters_used: Number of quarters in calculation
        - quality_score: 0-100 score
        - scaling_applied: True if fewer than 4 quarters were scaled
        - warnings: List of quality warnings

    Example:
        >>> quarters = get_rolling_ttm_periods(all_periods, num_quarters=4)
        >>> result = calculate_ttm_weighted(quarters)
        >>> print(f"TTM FCF: ${result.ttm_data.get('free_cash_flow', 0)/1e9:.2f}B")
        >>> print(f"Quality: {result.quality_score:.0f}/100")
    """
    warnings = []

    if not quarters:
        return TTMResult(
            ttm_data={},
            quarters_used=0,
            quality_score=0.0,
            scaling_applied=False,
            warnings=["No quarters provided"]
        )

    num_quarters = min(4, len(quarters))

    # Default to equal weights if not specified
    if weights is None:
        weights = [1.0] * num_quarters
    else:
        weights = weights[:num_quarters]

    # Normalize weights
    weight_sum = sum(weights)
    if weight_sum > 0:
        weights = [w / weight_sum * num_quarters for w in weights]

    # Collect metrics from all quarters
    ttm_data = {}
    metric_counts = {}  # Track how many quarters have each metric

    # Key metrics to aggregate
    aggregate_keys = [
        'operating_cash_flow', 'free_cash_flow', 'capital_expenditures',
        'total_revenue', 'net_income', 'operating_income', 'gross_profit',
        'dividends_paid', 'financing_cash_flow', 'investing_cash_flow'
    ]

    for i, quarter in enumerate(quarters[:num_quarters]):
        weight = weights[i] if i < len(weights) else 1.0

        # Process cash_flow metrics
        cf = quarter.get('cash_flow', {})
        for key in aggregate_keys:
            if key in cf and isinstance(cf[key], (int, float)):
                val = cf[key] * weight
                ttm_data[key] = ttm_data.get(key, 0) + val
                metric_counts[key] = metric_counts.get(key, 0) + 1

        # Process income_statement metrics
        income = quarter.get('income_statement', {})
        for key in aggregate_keys:
            if key in income and isinstance(income[key], (int, float)):
                val = income[key] * weight
                ttm_data[key] = ttm_data.get(key, 0) + val
                metric_counts[key] = metric_counts.get(key, 0) + 1

    # Scale to 4-quarter equivalent if fewer than 4 quarters
    scaling_applied = False
    if num_quarters < 4:
        scaling_factor = 4.0 / num_quarters
        for key in ttm_data:
            ttm_data[key] *= scaling_factor
        scaling_applied = True
        warnings.append(
            f"TTM scaled from {num_quarters} quarters to 4-quarter equivalent "
            f"(scaling factor: {scaling_factor:.2f}x)"
        )

    # Calculate quality score
    quality_score = _calculate_ttm_quality_score(
        num_quarters, metric_counts, scaling_applied
    )

    # Add quality warnings
    if quality_score < 70:
        warnings.append(f"Low quality TTM calculation (score: {quality_score:.0f}/100)")

    for key, count in metric_counts.items():
        if count < num_quarters:
            warnings.append(f"Metric '{key}' missing in {num_quarters - count} quarter(s)")

    return TTMResult(
        ttm_data=ttm_data,
        quarters_used=num_quarters,
        quality_score=quality_score,
        scaling_applied=scaling_applied,
        warnings=warnings
    )


def _calculate_ttm_quality_score(
    quarters_used: int,
    metric_counts: Dict[str, int],
    scaling_applied: bool
) -> float:
    """Calculate quality score for TTM calculation."""
    # Base score from quarters used
    quarter_score = quarters_used / 4.0 * 100  # 25 per quarter

    # Penalty for scaling
    scaling_penalty = 15 if scaling_applied else 0

    # Metric completeness (average across metrics)
    if metric_counts:
        completeness = sum(c / 4.0 for c in metric_counts.values()) / len(metric_counts)
        completeness_score = completeness * 100
    else:
        completeness_score = 0

    # Weighted average: 50% quarters, 30% completeness, 20% no-scaling
    quality = (
        0.50 * quarter_score +
        0.30 * completeness_score +
        0.20 * (100 - scaling_penalty * 5)
    )

    return max(0, min(100, quality))


def validate_computed_quarter(
    computed: Dict[str, Any],
    fy_data: Dict[str, Any],
    reported_quarters: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Validate computed Q4 against business logic rules.

    Checks:
    - No negative values for revenue, net_income, operating_cash_flow
    - Q4 values are reasonable (not > FY or < 0)
    - Sum of Q1+Q2+Q3+Q4 ≈ FY (within 1% tolerance)

    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []

    # Check for negative values in metrics that shouldn't be negative
    non_negative_keys = ['revenue', 'operating_cash_flow']
    for key in non_negative_keys:
        val = computed.get(key, 0)
        if val and val < 0:
            warnings.append(f"Computed {key} is negative: {val/1e6:.1f}M")

    # Check Q4 < FY for all metrics
    for key in ['revenue', 'operating_cash_flow', 'net_income']:
        q4_val = computed.get(key, 0)
        fy_val = extract_nested_value(fy_data, key)
        if q4_val and fy_val and q4_val > fy_val:
            warnings.append(f"Computed Q4 {key} ({q4_val/1e6:.1f}M) > FY ({fy_val/1e6:.1f}M)")

    # Validate sum: Q1+Q2+Q3+Q4 ≈ FY
    for key in ['operating_cash_flow', 'revenue']:
        q4_val = computed.get(key, 0)
        fy_val = extract_nested_value(fy_data, key)

        if q4_val and fy_val:
            quarters_sum = q4_val  # Start with Q4
            for q in reported_quarters:
                q_val = extract_nested_value(q, key)
                if q_val:
                    quarters_sum += q_val

            # Check if sum matches FY within 1%
            if fy_val > 0:
                diff_pct = abs(quarters_sum - fy_val) / fy_val * 100
                if diff_pct > 1.0:
                    warnings.append(
                        f"Sum mismatch for {key}: Q1+Q2+Q3+Q4={quarters_sum/1e6:.1f}M "
                        f"vs FY={fy_val/1e6:.1f}M (diff: {diff_pct:.1f}%)"
                    )

    is_valid = len(warnings) == 0
    return is_valid, warnings
