#!/usr/bin/env python3
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

"""Short Interest CLI.

Command-line interface for FINRA short interest and short volume data.

Usage:
    python scripts/short_interest_cli.py GME --current
    python scripts/short_interest_cli.py AMC --history --periods 12
    python scripts/short_interest_cli.py TSLA --squeeze
    python scripts/short_interest_cli.py --most-shorted --limit 25

Examples:
    # Get current short interest
    python scripts/short_interest_cli.py GME --current

    # Get short interest history
    python scripts/short_interest_cli.py AMC --history --periods 12

    # Get daily short volume
    python scripts/short_interest_cli.py AAPL --volume --days 30

    # Calculate squeeze risk
    python scripts/short_interest_cli.py GME --squeeze

    # Get most shorted stocks
    python scripts/short_interest_cli.py --most-shorted --limit 25
"""

import argparse
import asyncio
import json
import sys

# Add project root to path
sys.path.insert(0, "/Users/vijaysingh/code/victor-invest")


def format_current_output(data: dict) -> str:
    """Format current short interest for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 65}")
    lines.append(f"SHORT INTEREST: {data.get('symbol', 'N/A')}")
    lines.append(f"{'=' * 65}")

    if data.get("message"):
        lines.append(f"\n  {data['message']}")
        lines.append(f"\n{'=' * 65}")
        return "\n".join(lines)

    lines.append(f"\nSettlement Date: {data.get('settlement_date', 'N/A')}")

    lines.append(f"\nSHORT INTEREST METRICS")
    short_interest = data.get('short_interest', 0)
    if short_interest >= 1_000_000:
        lines.append(f"  Short Interest:       {short_interest / 1_000_000:.2f}M shares")
    else:
        lines.append(f"  Short Interest:       {short_interest:,} shares")

    avg_vol = data.get('avg_daily_volume', 0)
    if avg_vol >= 1_000_000:
        lines.append(f"  Avg Daily Volume:     {avg_vol / 1_000_000:.2f}M shares")
    else:
        lines.append(f"  Avg Daily Volume:     {avg_vol:,} shares")

    lines.append(f"  Days to Cover:        {data.get('days_to_cover', 0):.2f} days")

    if data.get('short_percent_float'):
        lines.append(f"  Short % of Float:     {data['short_percent_float']:.2f}%")
    if data.get('short_percent_outstanding'):
        lines.append(f"  Short % Outstanding:  {data['short_percent_outstanding']:.2f}%")

    # Previous period change
    prev = data.get('previous')
    if prev:
        lines.append(f"\nCHANGE FROM PREVIOUS")
        change = prev.get('change', 0)
        change_pct = prev.get('change_percent', 0)
        direction = "+" if change >= 0 else ""
        if abs(change) >= 1_000_000:
            lines.append(f"  Change (shares):      {direction}{change / 1_000_000:.2f}M")
        else:
            lines.append(f"  Change (shares):      {direction}{change:,}")
        lines.append(f"  Change (%):           {direction}{change_pct:.2f}%")

    # Investment signal
    signal = data.get('signal', {})
    if signal:
        lines.append(f"\nINVESTMENT SIGNAL")
        level = signal.get('level', 'neutral').upper().replace('_', ' ')
        lines.append(f"  Signal Level:         {level}")
        if signal.get('interpretation'):
            # Wrap interpretation text
            interp = signal['interpretation']
            lines.append(f"\n  {interp[:60]}")
            if len(interp) > 60:
                lines.append(f"  {interp[60:120]}")
            if len(interp) > 120:
                lines.append(f"  {interp[120:]}")

        if signal.get('factors'):
            lines.append(f"\n  Factors:")
            for factor in signal['factors']:
                lines.append(f"    - {factor}")

    lines.append(f"\n{'=' * 65}")
    return "\n".join(lines)


def format_history_output(data: dict) -> str:
    """Format short interest history for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 75}")
    lines.append(f"SHORT INTEREST HISTORY: {data.get('symbol', 'N/A')}")
    lines.append(f"{'=' * 75}")
    lines.append(f"\nPeriods: {data.get('periods', 0)}")

    # Trend analysis
    trend = data.get('trend_analysis', {})
    if trend:
        lines.append(f"\nTREND ANALYSIS")
        lines.append(f"  Direction:            {trend.get('direction', 'unknown').upper().replace('_', ' ')}")
        lines.append(f"  Avg Change/Period:    {trend.get('avg_change_per_period', 0):+.2f}%")
        lines.append(f"  Total Change:         {trend.get('total_change_pct', 0):+.2f}%")
        lines.append(f"  Periods Increasing:   {trend.get('periods_increasing', 0)}")
        lines.append(f"  Periods Decreasing:   {trend.get('periods_decreasing', 0)}")
        if trend.get('interpretation'):
            lines.append(f"\n  {trend['interpretation'][:70]}")
            if len(trend['interpretation']) > 70:
                lines.append(f"  {trend['interpretation'][70:]}")

    # History table
    history = data.get('history', [])
    if history:
        lines.append(f"\nHISTORICAL DATA")
        lines.append(f"  {'Settlement':<12} {'Short Interest':>15} {'Days Cover':>12} {'% Float':>10} {'Change':>10}")
        lines.append(f"  {'-' * 63}")
        for h in history[:12]:  # Limit to 12 most recent
            settle = h.get('settlement_date', 'N/A')[:10]
            si = h.get('short_interest', 0)
            if si >= 1_000_000:
                si_str = f"{si / 1_000_000:.2f}M"
            else:
                si_str = f"{si:,}"
            dtc = h.get('days_to_cover', 0)
            spf = h.get('short_percent_float')
            spf_str = f"{spf:.1f}%" if spf else "N/A"
            prev = h.get('previous')
            change_pct = prev.get('change_percent') if prev else None
            change_str = f"{change_pct:+.1f}%" if change_pct is not None else "N/A"
            lines.append(f"  {settle:<12} {si_str:>15} {dtc:>12.2f} {spf_str:>10} {change_str:>10}")
    else:
        lines.append("\n  No historical data available.")

    lines.append(f"\n{'=' * 75}")
    return "\n".join(lines)


def format_volume_output(data: dict) -> str:
    """Format short volume data for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"DAILY SHORT VOLUME: {data.get('symbol', 'N/A')}")
    lines.append(f"{'=' * 70}")
    lines.append(f"\nDays Analyzed: {data.get('days', 0)}")
    lines.append(f"Avg Short %:   {data.get('avg_short_percent', 0):.2f}%")

    volume = data.get('volume', [])
    if volume:
        lines.append(f"\nDAILY BREAKDOWN")
        lines.append(f"  {'Date':<12} {'Short Vol':>15} {'Total Vol':>15} {'Short %':>10}")
        lines.append(f"  {'-' * 55}")
        for v in volume[:15]:  # Limit to 15 most recent
            trade_date = v.get('trade_date', 'N/A')[:10]
            short_vol = v.get('short_volume', 0)
            total_vol = v.get('total_volume', 0)
            short_pct = v.get('short_percent', 0)
            if short_vol >= 1_000_000:
                sv_str = f"{short_vol / 1_000_000:.2f}M"
            else:
                sv_str = f"{short_vol:,}"
            if total_vol >= 1_000_000:
                tv_str = f"{total_vol / 1_000_000:.2f}M"
            else:
                tv_str = f"{total_vol:,}"
            lines.append(f"  {trade_date:<12} {sv_str:>15} {tv_str:>15} {short_pct:>9.1f}%")
    else:
        lines.append("\n  No short volume data available.")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_squeeze_output(data: dict) -> str:
    """Format squeeze risk assessment for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"SHORT SQUEEZE RISK: {data.get('symbol', 'N/A')}")
    lines.append(f"{'=' * 70}")

    # Risk score bar
    score = data.get('squeeze_score', 0)
    bar_len = 40
    filled = int(score / 100 * bar_len)
    bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'

    lines.append(f"\nSQUEEZE SCORE: {score:.1f}/100")
    lines.append(f"  {bar}")

    risk_level = data.get('risk_level', 'unknown').upper()
    lines.append(f"\nRISK LEVEL: {risk_level}")

    # Color code interpretation
    if risk_level == "EXTREME":
        lines.append("  [!!!] EXTREME SQUEEZE POTENTIAL")
    elif risk_level == "HIGH":
        lines.append("  [!!] HIGH SQUEEZE POTENTIAL")
    elif risk_level == "ELEVATED":
        lines.append("  [!] ELEVATED SQUEEZE POTENTIAL")
    elif risk_level == "MODERATE":
        lines.append("  MODERATE SQUEEZE POTENTIAL")
    else:
        lines.append("  LOW SQUEEZE POTENTIAL")

    # Factors
    factors = data.get('factors', [])
    if factors:
        lines.append(f"\nRISK FACTORS:")
        for factor in factors:
            lines.append(f"  - {factor}")

    # Interpretation
    interpretation = data.get('interpretation', '')
    if interpretation:
        lines.append(f"\nASSESSMENT:")
        # Wrap long text
        words = interpretation.split()
        line = "  "
        for word in words:
            if len(line) + len(word) > 68:
                lines.append(line)
                line = "  "
            line += word + " "
        if line.strip():
            lines.append(line)

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_most_shorted_output(data: dict) -> str:
    """Format most shorted stocks for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 75}")
    lines.append(f"MOST SHORTED STOCKS")
    lines.append(f"{'=' * 75}")
    lines.append(f"\nTotal: {data.get('count', 0)} stocks")

    stocks = data.get('stocks', [])
    if stocks:
        lines.append(f"\n{'#':<4} {'Symbol':<8} {'% Float':>10} {'Days Cover':>12} {'Short Interest':>18}")
        lines.append(f"{'-' * 55}")
        for i, s in enumerate(stocks, 1):
            symbol = s.get('symbol', 'N/A')
            spf = s.get('short_percent_float', 0)
            dtc = s.get('days_to_cover', 0)
            si = s.get('short_interest', 0)
            if si >= 1_000_000_000:
                si_str = f"{si / 1_000_000_000:.2f}B"
            elif si >= 1_000_000:
                si_str = f"{si / 1_000_000:.2f}M"
            else:
                si_str = f"{si:,}"
            lines.append(f"{i:<4} {symbol:<8} {spf:>9.2f}% {dtc:>12.2f} {si_str:>18}")
    else:
        lines.append("\n  No data available.")

    lines.append(f"\n{'=' * 75}")
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="FINRA Short Interest and Short Volume CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s GME --current                Get current short interest
  %(prog)s AMC --history --periods 12   Get 12 periods of history
  %(prog)s AAPL --volume --days 30      Get 30 days of short volume
  %(prog)s GME --squeeze                Calculate squeeze risk
  %(prog)s --most-shorted --limit 25    Get 25 most shorted stocks
        """
    )

    # Symbol argument (positional, optional)
    parser.add_argument(
        "symbol",
        nargs="?",
        help="Stock ticker symbol"
    )

    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--current",
        action="store_true",
        help="Get current short interest"
    )
    action_group.add_argument(
        "--history",
        action="store_true",
        help="Get historical short interest"
    )
    action_group.add_argument(
        "--volume",
        action="store_true",
        help="Get daily short volume"
    )
    action_group.add_argument(
        "--squeeze",
        action="store_true",
        help="Calculate short squeeze risk"
    )
    action_group.add_argument(
        "--most-shorted",
        action="store_true",
        help="Get most shorted stocks"
    )

    # Options
    parser.add_argument(
        "--periods",
        type=int,
        default=12,
        help="Number of bi-monthly periods for history (default: 12)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of trading days for volume (default: 30)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of stocks for most-shorted (default: 20)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text"
    )

    args = parser.parse_args()

    # Validate symbol is provided for symbol-based actions
    if (args.current or args.history or args.volume or args.squeeze) and not args.symbol:
        parser.error("Symbol is required for --current, --history, --volume, and --squeeze")

    # Import and initialize tool
    from victor_invest.tools.short_interest import ShortInterestTool

    tool = ShortInterestTool()
    await tool.initialize()

    # Determine action and execute
    if args.current:
        result = await tool.execute(
            action="current",
            symbol=args.symbol
        )
        formatter = format_current_output
    elif args.history:
        result = await tool.execute(
            action="history",
            symbol=args.symbol,
            periods=args.periods
        )
        formatter = format_history_output
    elif args.volume:
        result = await tool.execute(
            action="volume",
            symbol=args.symbol,
            days=args.days
        )
        formatter = format_volume_output
    elif args.squeeze:
        result = await tool.execute(
            action="squeeze",
            symbol=args.symbol
        )
        formatter = format_squeeze_output
    elif args.most_shorted:
        result = await tool.execute(
            action="most_shorted",
            limit=args.limit
        )
        formatter = format_most_shorted_output
    else:
        print("Error: No action specified", file=sys.stderr)
        sys.exit(1)

    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        print(json.dumps(result.data, indent=2))
    else:
        print(formatter(result.data))


if __name__ == "__main__":
    asyncio.run(main())
