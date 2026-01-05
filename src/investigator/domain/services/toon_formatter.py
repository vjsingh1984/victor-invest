#!/usr/bin/env python3
"""
TOON (Token-Oriented Object Notation) Formatter
Optimizes token usage for LLM prompts by converting tabular JSON to TOON format.

Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0
"""

import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class TOONFormatter:
    """
    Converts tabular JSON data to TOON format for token-efficient LLM prompts.

    TOON saves 30-60% tokens on uniform tabular data by:
    1. Declaring column headers once
    2. Streaming values in rows without repeating keys
    3. Minimizing punctuation (no {}, [], :, per row)

    Example:
        JSON (85 tokens):
        [
          {"year": 2024, "quarter": "Q3", "revenue": 5.5B},
          {"year": 2024, "quarter": "Q2", "revenue": 5.3B}
        ]

        TOON (42 tokens):
        data[2]{year,quarter,revenue}:
        2024,Q3,5.5B
        2024,Q2,5.3B
    """

    @staticmethod
    def format_array(data: List[Dict[str, Any]], name: str = "data", precision: int = 2) -> str:
        """
        Convert uniform JSON array to TOON format.

        Args:
            data: List of dicts with identical keys
            name: Name for the dataset
            precision: Decimal precision for floats

        Returns:
            TOON-formatted string
        """
        if not data:
            return f"{name}[0]"

        # Extract column names from first row
        columns = list(data[0].keys())
        count = len(data)

        # Header: name[count]{col1,col2,...}:
        header = f"{name}[{count}]{{{','.join(columns)}}}:"

        # Rows: val1,val2,val3,...
        rows = []
        for row in data:
            formatted_values = []
            for col in columns:
                val = row.get(col)
                formatted_val = TOONFormatter._format_value(val, precision)
                formatted_values.append(formatted_val)
            rows.append(",".join(formatted_values))

        return header + "\n" + "\n".join(rows)

    @staticmethod
    def _format_value(val: Any, precision: int = 2) -> str:
        """Format a single value for TOON output."""
        if val is None:
            return "null"
        elif isinstance(val, bool):
            return "true" if val else "false"
        elif isinstance(val, float):
            return f"{val:.{precision}f}"
        elif isinstance(val, (int, str)):
            # Escape commas in strings
            if isinstance(val, str) and "," in val:
                return f'"{val}"'
            return str(val)
        else:
            # Complex types as JSON string
            import json

            return json.dumps(val)

    @staticmethod
    def format_quarterly_data(quarters: List[Dict[str, Any]], metric_keys: List[str] = None) -> str:
        """
        Format quarterly financial data with smart metric selection.

        Args:
            quarters: List of quarterly data dicts
            metric_keys: Optional list of specific metrics to include

        Returns:
            TOON-formatted quarterly data
        """
        if not quarters:
            return "quarterly_data[0]"

        # If metric_keys not provided, use all keys from first quarter
        if metric_keys is None:
            # Prioritize key metrics first
            priority_keys = [
                "fiscal_year",
                "fiscal_period",
                "period_end_date",
                "revenue",
                "net_income",
                "operating_cash_flow",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "stockholders_equity",
                "earnings_per_share",
                "shares_outstanding",
            ]
            all_keys = list(quarters[0].keys())

            # Priority keys that exist + remaining keys
            metric_keys = [k for k in priority_keys if k in all_keys]
            metric_keys += [k for k in all_keys if k not in metric_keys]

        # Filter quarters to only include specified keys
        filtered_quarters = [{k: q.get(k) for k in metric_keys if k in q} for q in quarters]

        return TOONFormatter.format_array(filtered_quarters, name="quarterly_data", precision=2)

    @staticmethod
    def format_peer_comparison(peers: List[Dict[str, Any]], metric_keys: List[str] = None) -> str:
        """
        Format peer comparison data.

        Args:
            peers: List of peer company data
            metric_keys: Optional list of specific metrics

        Returns:
            TOON-formatted peer data
        """
        if metric_keys:
            filtered_peers = [{k: p.get(k) for k in metric_keys if k in p} for p in peers]
        else:
            filtered_peers = peers

        return TOONFormatter.format_array(filtered_peers, name="peer_companies", precision=2)

    @staticmethod
    def get_format_explanation() -> str:
        """
        Get TOON format explanation for system prompts.

        Returns:
            Explanation string to add to LLM system prompt
        """
        return """
# Data Format: TOON (Token-Oriented Object Notation)

Financial data is provided in TOON format for efficiency:

Format: `dataset_name[count]{column1,column2,...}:`
Followed by rows of comma-separated values.

Example:
```
quarterly_data[4]{fiscal_year,fiscal_period,revenue,net_income}:
2024,Q3,5500000000,531000000
2024,Q2,5300000000,530000000
2024,Q1,5530000000,553000000
2023,Q4,5100000000,490000000
```

- First line: Header with column names
- Following lines: Data rows (same order as header)
- null: Missing/unavailable data
- Numbers: Raw values (dollars for financials)
"""


# Convenience functions
def to_toon_quarterly(quarters: List[Dict[str, Any]], metrics: List[str] = None) -> str:
    """Convert quarterly data to TOON format."""
    return TOONFormatter.format_quarterly_data(quarters, metrics)


def to_toon_peers(peers: List[Dict[str, Any]], metrics: List[str] = None) -> str:
    """Convert peer comparison data to TOON format."""
    return TOONFormatter.format_peer_comparison(peers, metrics)


def to_toon_array(data: List[Dict[str, Any]], name: str = "data") -> str:
    """Convert any uniform array to TOON format."""
    return TOONFormatter.format_array(data, name)
