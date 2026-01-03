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

"""Institutional Holdings CLI.

Command-line interface for SEC Form 13F institutional holdings data.

Usage:
    python scripts/institutional_holdings_cli.py AAPL --holdings
    python scripts/institutional_holdings_cli.py AAPL --top-holders --limit 20
    python scripts/institutional_holdings_cli.py AAPL --changes --quarters 4
    python scripts/institutional_holdings_cli.py --institution 0001067983
    python scripts/institutional_holdings_cli.py --search "berkshire"

Examples:
    # Get institutional holdings for AAPL
    python scripts/institutional_holdings_cli.py AAPL --holdings

    # Get top 20 institutional holders
    python scripts/institutional_holdings_cli.py AAPL --top-holders --limit 20

    # Get ownership changes over 4 quarters
    python scripts/institutional_holdings_cli.py AAPL --changes --quarters 4

    # Get holdings for Berkshire Hathaway
    python scripts/institutional_holdings_cli.py --institution 0001067983

    # Search for institutions
    python scripts/institutional_holdings_cli.py --search "vanguard"
"""

import argparse
import asyncio
import json
import sys

# Add project root to path
sys.path.insert(0, "/Users/vijaysingh/code/victor-invest")


def format_holdings_output(data: dict) -> str:
    """Format holdings data for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 65}")
    lines.append(f"INSTITUTIONAL HOLDINGS: {data.get('symbol', 'N/A')}")
    lines.append(f"{'=' * 65}")
    lines.append(f"\nReport Quarter: {data.get('report_quarter', 'N/A')}")

    lines.append(f"\nOWNERSHIP SUMMARY")
    lines.append(f"  Total Institutions:   {data.get('num_institutions', 0):,}")
    lines.append(f"  Total Shares:         {data.get('total_shares', 0):,}")

    total_value = data.get('total_value_dollars', 0)
    if total_value >= 1_000_000_000:
        lines.append(f"  Total Value:          ${total_value / 1_000_000_000:.2f}B")
    elif total_value >= 1_000_000:
        lines.append(f"  Total Value:          ${total_value / 1_000_000:.2f}M")
    else:
        lines.append(f"  Total Value:          ${total_value:,.0f}")

    if data.get('ownership_pct'):
        lines.append(f"  Ownership %:          {data.get('ownership_pct'):.2f}%")

    # Changes
    changes = data.get('changes', {})
    if changes.get('qoq_change_pct') is not None:
        lines.append(f"\nQUARTER-OVER-QUARTER CHANGE")
        qoq_pct = changes.get('qoq_change_pct', 0)
        qoq_shares = changes.get('qoq_change_shares', 0)
        direction = "+" if qoq_pct >= 0 else ""
        lines.append(f"  Change %:             {direction}{qoq_pct:.2f}%")
        direction = "+" if qoq_shares >= 0 else ""
        lines.append(f"  Change Shares:        {direction}{qoq_shares:,}")

    # Investment signal
    signal = data.get('investment_signal', {})
    if signal:
        lines.append(f"\nINVESTMENT SIGNAL")
        level = signal.get('level', 'neutral').upper().replace('_', ' ')
        lines.append(f"  Signal:               {level}")
        if signal.get('interpretation'):
            lines.append(f"  Interpretation:       {signal['interpretation']}")
        for factor in signal.get('factors', []):
            lines.append(f"    - {factor}")

    # Top holders
    top_holders = data.get('top_holders', [])
    if top_holders:
        lines.append(f"\nTOP INSTITUTIONAL HOLDERS")
        lines.append(f"  {'#':<3} {'Institution':<35} {'Value':>12} {'Shares':>15}")
        lines.append(f"  {'-' * 68}")
        for i, holder in enumerate(top_holders[:10], 1):
            name = holder.get('name', 'Unknown')[:32]
            value = holder.get('value_thousands', 0) * 1000
            shares = holder.get('shares', 0)
            if value >= 1_000_000_000:
                value_str = f"${value / 1_000_000_000:.1f}B"
            elif value >= 1_000_000:
                value_str = f"${value / 1_000_000:.1f}M"
            else:
                value_str = f"${value:,.0f}"
            lines.append(f"  {i:<3} {name:<35} {value_str:>12} {shares:>15,}")

    lines.append(f"\n{'=' * 65}")
    return "\n".join(lines)


def format_top_holders_output(data: dict) -> str:
    """Format top holders data for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 75}")
    lines.append(f"TOP INSTITUTIONAL HOLDERS: {data.get('symbol', 'N/A')}")
    lines.append(f"{'=' * 75}")

    lines.append(f"\nTotal Holders Found: {data.get('num_holders', 0)}")
    total_value = data.get('total_value_dollars', 0)
    if total_value >= 1_000_000_000:
        lines.append(f"Total Value (shown): ${total_value / 1_000_000_000:.2f}B")
    elif total_value >= 1_000_000:
        lines.append(f"Total Value (shown): ${total_value / 1_000_000:.2f}M")
    else:
        lines.append(f"Total Value (shown): ${total_value:,.0f}")

    top_holders = data.get('top_holders', [])
    if top_holders:
        lines.append(f"\n{'#':<3} {'Institution':<40} {'Value':>15} {'Shares':>18}")
        lines.append(f"{'-' * 78}")
        for i, holder in enumerate(top_holders, 1):
            name = holder.get('name', 'Unknown')[:37]
            value = holder.get('value_thousands', 0) * 1000
            shares = holder.get('shares', 0)
            if value >= 1_000_000_000:
                value_str = f"${value / 1_000_000_000:.2f}B"
            elif value >= 1_000_000:
                value_str = f"${value / 1_000_000:.2f}M"
            else:
                value_str = f"${value:,.0f}"
            lines.append(f"{i:<3} {name:<40} {value_str:>15} {shares:>18,}")
    else:
        lines.append("\n  No institutional holders found.")

    lines.append(f"\n{'=' * 75}")
    return "\n".join(lines)


def format_changes_output(data: dict) -> str:
    """Format ownership changes for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"INSTITUTIONAL OWNERSHIP CHANGES: {data.get('symbol', 'N/A')}")
    lines.append(f"{'=' * 70}")
    lines.append(f"\nQuarters Analyzed: {data.get('quarters_analyzed', 0)}")

    # Trend analysis
    trend = data.get('trend_analysis', {})
    if trend:
        lines.append(f"\nTREND ANALYSIS")
        lines.append(f"  Direction:            {trend.get('direction', 'unknown').upper()}")
        lines.append(f"  Avg Quarterly Change: {trend.get('avg_quarterly_change_pct', 0):+.2f}%")
        lines.append(f"  Total Period Change:  {trend.get('total_change_pct', 0):+.2f}%")
        lines.append(f"  Up Quarters:          {trend.get('up_quarters', 0)}")
        lines.append(f"  Down Quarters:        {trend.get('down_quarters', 0)}")
        if trend.get('interpretation'):
            lines.append(f"\n  {trend['interpretation']}")

    # Quarterly data
    changes = data.get('changes', [])
    if changes:
        lines.append(f"\nQUARTERLY BREAKDOWN")
        lines.append(f"  {'Quarter':<12} {'Institutions':>12} {'Shares':>18} {'Change':>12}")
        lines.append(f"  {'-' * 58}")
        for change in changes:
            quarter = change.get('quarter', 'N/A')[:12]
            num_inst = change.get('num_institutions', 0)
            shares = change.get('total_shares', 0)
            qoq = change.get('qoq_change_pct')
            qoq_str = f"{qoq:+.1f}%" if qoq is not None else "N/A"
            lines.append(f"  {quarter:<12} {num_inst:>12,} {shares:>18,} {qoq_str:>12}")
    else:
        lines.append("\n  No historical data available.")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_institution_output(data: dict) -> str:
    """Format institution holdings for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"INSTITUTION HOLDINGS: CIK {data.get('cik', 'N/A')}")
    lines.append(f"{'=' * 80}")

    lines.append(f"\nTotal Positions: {data.get('total_positions', 0)}")
    total_value = data.get('total_value_dollars', 0)
    if total_value >= 1_000_000_000:
        lines.append(f"Total Value: ${total_value / 1_000_000_000:.2f}B")
    elif total_value >= 1_000_000:
        lines.append(f"Total Value: ${total_value / 1_000_000:.2f}M")
    else:
        lines.append(f"Total Value: ${total_value:,.0f}")

    holdings = data.get('top_holdings', [])
    if holdings:
        lines.append(f"\nTOP HOLDINGS")
        lines.append(f"{'#':<3} {'Symbol':<8} {'Issuer':<30} {'Value':>15} {'Shares':>15}")
        lines.append(f"{'-' * 75}")
        for i, h in enumerate(holdings[:25], 1):
            symbol = (h.get('symbol') or h.get('cusip', 'N/A'))[:7]
            issuer = h.get('issuer_name', 'Unknown')[:28]
            value = h.get('value_dollars', 0)
            shares = h.get('shares', 0)
            if value >= 1_000_000_000:
                value_str = f"${value / 1_000_000_000:.2f}B"
            elif value >= 1_000_000:
                value_str = f"${value / 1_000_000:.2f}M"
            else:
                value_str = f"${value:,.0f}"
            lines.append(f"{i:<3} {symbol:<8} {issuer:<30} {value_str:>15} {shares:>15,}")
    else:
        lines.append("\n  No holdings found for this institution.")

    lines.append(f"\n{'=' * 80}")
    return "\n".join(lines)


def format_search_output(data: dict) -> str:
    """Format institution search results for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"INSTITUTION SEARCH: \"{data.get('query', '')}\"")
    lines.append(f"{'=' * 70}")
    lines.append(f"\nResults Found: {data.get('num_results', 0)}")

    institutions = data.get('institutions', [])
    if institutions:
        lines.append(f"\n{'CIK':<12} {'Institution Name':<45} {'Latest Filing':<15}")
        lines.append(f"{'-' * 72}")
        for inst in institutions:
            cik = inst.get('cik', 'N/A')[:11]
            name = inst.get('name', 'Unknown')[:43]
            filing = inst.get('latest_filing', 'N/A')[:14]
            lines.append(f"{cik:<12} {name:<45} {filing:<15}")

        lines.append(f"\nUse --institution <CIK> to view holdings for a specific institution")
    else:
        lines.append("\n  No institutions found matching your query.")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="SEC Form 13F Institutional Holdings CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL --holdings          Get institutional holdings for AAPL
  %(prog)s AAPL --top-holders       Get top institutional holders
  %(prog)s AAPL --changes           Get ownership changes over time
  %(prog)s --institution 0001067983 Get holdings for Berkshire Hathaway
  %(prog)s --search "vanguard"      Search for institutions
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
        "--holdings",
        action="store_true",
        help="Get institutional holdings for symbol"
    )
    action_group.add_argument(
        "--top-holders",
        action="store_true",
        help="Get top institutional holders"
    )
    action_group.add_argument(
        "--changes",
        action="store_true",
        help="Get ownership changes over time"
    )
    action_group.add_argument(
        "--institution",
        metavar="CIK",
        help="Get holdings for a specific institution by CIK"
    )
    action_group.add_argument(
        "--search",
        metavar="QUERY",
        help="Search for institutions by name"
    )

    # Options
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of results to return (default: 20)"
    )
    parser.add_argument(
        "--quarters",
        type=int,
        default=4,
        help="Number of quarters for change analysis (default: 4)"
    )
    parser.add_argument(
        "--quarter",
        type=str,
        help="Specific quarter (e.g., '2024-Q4')"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text"
    )

    args = parser.parse_args()

    # Validate symbol is provided for symbol-based actions
    if (args.holdings or args.top_holders or args.changes) and not args.symbol:
        parser.error("Symbol is required for --holdings, --top-holders, and --changes")

    # Import and initialize tool
    from victor_invest.tools.institutional_holdings import InstitutionalHoldingsTool

    tool = InstitutionalHoldingsTool()
    await tool.initialize()

    # Determine action and execute
    if args.holdings:
        result = await tool.execute(
            action="holdings",
            symbol=args.symbol,
            quarter=args.quarter
        )
        formatter = format_holdings_output
    elif args.top_holders:
        result = await tool.execute(
            action="top_holders",
            symbol=args.symbol,
            limit=args.limit
        )
        formatter = format_top_holders_output
    elif args.changes:
        result = await tool.execute(
            action="changes",
            symbol=args.symbol,
            quarters=args.quarters
        )
        formatter = format_changes_output
    elif args.institution:
        result = await tool.execute(
            action="institution",
            cik=args.institution,
            quarter=args.quarter
        )
        formatter = format_institution_output
    elif args.search:
        result = await tool.execute(
            action="search",
            query=args.search,
            limit=args.limit
        )
        formatter = format_search_output
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
