#!/usr/bin/env python3
"""
Macro Data CLI - Access FRED macroeconomic data via command line.

This script provides CLI access to the MacroDataTool which exposes
FRED (Federal Reserve Economic Data) stored in the PostgreSQL database.

Features:
- Category-based indicator queries (growth, inflation, rates, etc.)
- Specific indicator lookups by FRED series ID
- Buffett Indicator calculation (Market Cap / GDP)
- Time series historical data retrieval
- JSON and table output formats

Usage:
    # Get macro summary with all indicators
    python3 scripts/macro_data_cli.py --summary

    # Get Buffett Indicator
    python3 scripts/macro_data_cli.py --buffett

    # Get specific category
    python3 scripts/macro_data_cli.py --category rates

    # Get specific indicators
    python3 scripts/macro_data_cli.py --indicators DGS10 FEDFUNDS VIXCLS

    # Get time series
    python3 scripts/macro_data_cli.py --time-series DGS10 --limit 30

    # List available categories
    python3 scripts/macro_data_cli.py --list-categories

    # Output as JSON
    python3 scripts/macro_data_cli.py --summary --json

Author: Victor-Invest Team
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor_invest.tools import MacroDataTool

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_value(value: Any, units: Optional[str] = None) -> str:
    """Format a value for display based on its units."""
    if value is None:
        return "N/A"

    if isinstance(value, (int, float)):
        if units:
            units_lower = units.lower() if units else ""
            if "percent" in units_lower or "rate" in units_lower:
                return f"{value:.2f}%"
            elif "billions" in units_lower:
                return f"${value:,.1f}B"
            elif "millions" in units_lower:
                return f"${value:,.1f}M"
            elif "thousands" in units_lower:
                return f"{value:,.0f}K"
            elif "index" in units_lower:
                return f"{value:,.2f}"

        # Default number formatting
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        elif abs(value) >= 1:
            return f"{value:.2f}"
        else:
            return f"{value:.4f}"

    return str(value)


def format_change(change_data: Optional[Dict]) -> str:
    """Format change information for display."""
    if not change_data or change_data.get("percent") is None:
        return ""

    pct = change_data["percent"]
    arrow = "↑" if pct > 0 else "↓" if pct < 0 else "→"
    return f" {arrow} {abs(pct):.2f}%"


def print_indicator_table(indicators: Dict, title: str = "Indicators"):
    """Print indicators in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}")

    if not indicators:
        print("No data available")
        return

    # Header
    print(f"{'ID':<20} {'Name':<30} {'Value':<15} {'Change':<12} {'Date':<12}")
    print("-" * 80)

    for ind_id, data in sorted(indicators.items()):
        if data is None:
            continue

        name = (data.get("name", ind_id)[:28] + "..") if len(data.get("name", ind_id)) > 30 else data.get("name", ind_id)
        value_str = format_value(data.get("value"), data.get("units"))
        change_str = format_change(data.get("change"))
        date_str = data.get("date", "N/A")[:10] if data.get("date") else "N/A"

        print(f"{ind_id:<20} {name:<30} {value_str:<15} {change_str:<12} {date_str:<12}")


def print_buffett_indicator(data: Dict):
    """Print Buffett Indicator in formatted display."""
    print(f"\n{'=' * 80}")
    print(f"{'BUFFETT INDICATOR (Market Cap / GDP)':^80}")
    print(f"{'=' * 80}")

    ratio = data.get("ratio_percent")
    interpretation = data.get("interpretation", "N/A")
    signal = data.get("signal", "N/A")

    # Signal color codes for terminal
    signal_colors = {
        "strong_buy": "\033[92m",  # Green
        "buy": "\033[92m",
        "neutral": "\033[93m",  # Yellow
        "caution": "\033[93m",
        "warning": "\033[91m",  # Red
    }
    reset_color = "\033[0m"

    color = signal_colors.get(signal, "")

    print(f"\n  Ratio: {color}{ratio:.1f}%{reset_color}")
    print(f"  Interpretation: {color}{interpretation}{reset_color}")
    print(f"  Signal: {color}{signal.upper()}{reset_color}")

    # Components
    components = data.get("components", {})
    print("\n  Components:")
    print(f"    VTI Price: ${components.get('vti_price', 'N/A')}")
    print(f"    VTI Date: {components.get('vti_date', 'N/A')}")
    print(f"    GDP: ${components.get('gdp_billions', 'N/A'):,.0f}B" if components.get('gdp_billions') else "    GDP: N/A")
    print(f"    GDP Date: {components.get('gdp_date', 'N/A')}")
    print(f"    Est. Market Cap: ${components.get('estimated_market_cap_billions', 'N/A'):,.0f}B" if components.get('estimated_market_cap_billions') else "    Est. Market Cap: N/A")

    # Interpretation guide
    print("\n  Interpretation Guide:")
    for range_str, desc in data.get("interpretation_guide", {}).items():
        print(f"    {range_str}: {desc}")


def print_time_series(data: Dict):
    """Print time series data in formatted display."""
    print(f"\n{'=' * 80}")
    indicator_id = data.get("indicator_id", "Unknown")
    print(f"{'TIME SERIES: ' + indicator_id:^80}")
    print(f"{'=' * 80}")

    date_range = data.get("date_range", {})
    stats = data.get("statistics", {})
    time_series = data.get("time_series", [])

    print(f"\n  Data Points: {data.get('data_points', 0)}")
    print(f"  Date Range: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")

    print("\n  Statistics:")
    print(f"    Latest: {format_value(stats.get('latest'))}")
    print(f"    Min: {format_value(stats.get('min'))}")
    print(f"    Max: {format_value(stats.get('max'))}")
    print(f"    Mean: {format_value(stats.get('mean'))}")
    print(f"    Std Dev: {format_value(stats.get('std'))}")

    # Show recent values
    if time_series:
        print("\n  Recent Values (last 10):")
        print(f"  {'Date':<12} {'Value':<15}")
        print("  " + "-" * 27)
        for item in time_series[-10:]:
            print(f"  {item['date']:<12} {format_value(item['value']):<15}")


def print_categories(data: Dict):
    """Print available categories."""
    print(f"\n{'=' * 80}")
    print(f"{'AVAILABLE MACRO DATA CATEGORIES':^80}")
    print(f"{'=' * 80}")

    print(f"\n  Total Categories: {data.get('total_categories', 0)}")
    print(f"  Total Indicators: {data.get('total_indicators', 0)}")

    categories = data.get("categories", {})
    for cat_name, cat_info in sorted(categories.items()):
        print(f"\n  {cat_name.upper()} ({cat_info.get('indicator_count', 0)} indicators):")
        for ind in cat_info.get("indicators", []):
            print(f"    - {ind}")


def print_summary(data: Dict):
    """Print comprehensive macro summary."""
    print(f"\n{'=' * 80}")
    print(f"{'MACROECONOMIC SUMMARY':^80}")
    print(f"{'=' * 80}")

    timestamp = data.get("timestamp", datetime.now().isoformat())
    assessment = data.get("overall_assessment", "unknown")

    # Assessment color
    assessment_colors = {
        "favorable": "\033[92m",  # Green
        "mixed": "\033[93m",  # Yellow
        "cautionary": "\033[91m",  # Red
    }
    color = assessment_colors.get(assessment, "")
    reset = "\033[0m"

    print(f"\n  Timestamp: {timestamp}")
    print(f"  Overall Assessment: {color}{assessment.upper()}{reset}")

    # Alerts
    alerts = data.get("alerts", [])
    if alerts:
        print(f"\n  ALERTS ({len(alerts)}):")
        for alert in alerts:
            severity = alert.get("severity", "medium")
            severity_color = "\033[91m" if severity == "high" else "\033[93m"
            print(f"    {severity_color}[{severity.upper()}]{reset} {alert.get('indicator', 'N/A')}: {alert.get('type', 'N/A')}")

    # Buffett Indicator if present
    if "buffett_indicator" in data:
        bi = data["buffett_indicator"]
        print(f"\n  Buffett Indicator: {bi.get('ratio', 'N/A'):.1f}% - {bi.get('interpretation', 'N/A')}")

    # Categories
    categories = data.get("categories", {})
    for cat_name, indicators in sorted(categories.items()):
        if indicators:
            print_indicator_table(indicators, f"{cat_name.upper()} INDICATORS")


async def main():
    parser = argparse.ArgumentParser(
        description="Access FRED macroeconomic data via command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/macro_data_cli.py --summary
  python3 scripts/macro_data_cli.py --buffett
  python3 scripts/macro_data_cli.py --category rates
  python3 scripts/macro_data_cli.py --indicators DGS10 FEDFUNDS VIXCLS
  python3 scripts/macro_data_cli.py --time-series DGS10 --limit 30
  python3 scripts/macro_data_cli.py --list-categories
        """
    )

    # Actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--summary",
        action="store_true",
        help="Get comprehensive macro summary with all indicators"
    )
    action_group.add_argument(
        "--buffett",
        action="store_true",
        help="Get Buffett Indicator (Market Cap / GDP ratio)"
    )
    action_group.add_argument(
        "--category",
        type=str,
        metavar="NAME",
        help="Get indicators for category (growth, employment, inflation, rates, credit, debt, market, sentiment, housing, monetary, trade)"
    )
    action_group.add_argument(
        "--indicators",
        nargs="+",
        metavar="ID",
        help="Get specific indicators by FRED series ID"
    )
    action_group.add_argument(
        "--time-series",
        type=str,
        metavar="ID",
        help="Get historical time series for indicator"
    )
    action_group.add_argument(
        "--list-categories",
        action="store_true",
        help="List available categories and their indicators"
    )

    # Options
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=1095,
        help="Days of historical data for lookback (default: 1095 = 3 years)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max data points for time series (default: 1000)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted table"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Initialize tool
    tool = MacroDataTool()
    await tool.initialize()

    # Execute action
    result = None

    if args.summary:
        result = await tool.execute(action="get_summary")
    elif args.buffett:
        result = await tool.execute(action="buffett_indicator")
    elif args.category:
        result = await tool.execute(
            action="get_category",
            category=args.category,
            lookback_days=args.lookback_days
        )
    elif args.indicators:
        result = await tool.execute(
            action="get_indicators",
            indicators=args.indicators,
            lookback_days=args.lookback_days
        )
    elif args.time_series:
        result = await tool.execute(
            action="get_time_series",
            indicator_id=args.time_series,
            limit=args.limit
        )
    elif args.list_categories:
        result = await tool.execute(action="list_categories")

    # Handle result
    if result is None:
        print("Error: No action executed", file=sys.stderr)
        return 1

    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1

    # Output
    if args.json:
        print(json.dumps(result.data, indent=2, default=str))
    else:
        # Formatted output based on action
        if args.summary:
            print_summary(result.data)
        elif args.buffett:
            print_buffett_indicator(result.data)
        elif args.category:
            print_indicator_table(
                result.data.get("indicators", {}),
                f"{args.category.upper()} INDICATORS"
            )
        elif args.indicators:
            print_indicator_table(
                result.data.get("indicators", {}),
                "REQUESTED INDICATORS"
            )
            if result.warnings:
                print(f"\nWarnings: {result.warnings}")
        elif args.time_series:
            print_time_series(result.data)
        elif args.list_categories:
            print_categories(result.data)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
