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

"""Insider Trading CLI.

Command-line interface for SEC Form 4 insider trading analysis.

Usage:
    python scripts/insider_trading_cli.py AAPL --sentiment
    python scripts/insider_trading_cli.py AAPL --recent --days 30
    python scripts/insider_trading_cli.py AAPL --clusters
    python scripts/insider_trading_cli.py AAPL --key-insiders
    python scripts/insider_trading_cli.py AAPL --fetch

Examples:
    # Get insider sentiment for Apple
    python scripts/insider_trading_cli.py AAPL --sentiment

    # Get recent transactions for Microsoft (last 60 days)
    python scripts/insider_trading_cli.py MSFT --recent --days 60

    # Detect cluster buying/selling
    python scripts/insider_trading_cli.py NVDA --clusters

    # Get C-suite and director activity
    python scripts/insider_trading_cli.py TSLA --key-insiders
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, "/Users/vijaysingh/code/victor-invest")


def format_currency(value: float) -> str:
    """Format value as currency."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.0f}"


def format_sentiment_output(data: dict) -> str:
    """Format sentiment data for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"INSIDER SENTIMENT: {data['symbol']}")
    lines.append(f"{'=' * 60}")

    # Main sentiment score
    score = data.get('sentiment_score', 0)
    level = data.get('sentiment_level', 'no_data').upper()

    # Color-code sentiment level in description
    if 'bullish' in level.lower():
        indicator = "[+]"
    elif 'bearish' in level.lower():
        indicator = "[-]"
    else:
        indicator = "[=]"

    lines.append(f"\nSentiment: {indicator} {level} (score: {score:+.4f})")

    # Transaction summary
    txns = data.get('transactions', {})
    lines.append(f"\nTransactions ({data.get('metadata', {}).get('analysis_period_days', 90)} days):")
    lines.append(f"  Purchases: {txns.get('purchase_count', 0)}")
    lines.append(f"  Sales:     {txns.get('sale_count', 0)}")
    lines.append(f"  Total:     {txns.get('total_count', 0)}")

    # Value summary
    vals = data.get('values', {})
    lines.append(f"\nValues:")
    lines.append(f"  Purchase Value: {format_currency(vals.get('purchase_value', 0))}")
    lines.append(f"  Sale Value:     {format_currency(vals.get('sale_value', 0))}")
    lines.append(f"  Net Value:      {format_currency(vals.get('net_value', 0))}")

    # Insider summary
    insiders = data.get('insiders', {})
    lines.append(f"\nInsiders:")
    lines.append(f"  Unique Insiders:     {insiders.get('unique_count', 0)}")
    lines.append(f"  Key Insider Activity: {insiders.get('key_insider_activity', 0)}")

    # Signals
    signals = data.get('signals', {})
    lines.append(f"\nSignals:")
    lines.append(f"  Cluster Detected:    {'Yes' if signals.get('cluster_detected') else 'No'}")
    if signals.get('cluster_detected'):
        lines.append(f"  Cluster Type:        {signals.get('cluster_type', 'unknown')}")
    lines.append(f"  Significant Filings: {signals.get('significant_filings', 0)}")

    # Confidence
    meta = data.get('metadata', {})
    lines.append(f"\nConfidence: {meta.get('confidence', 0) * 100:.0f}%")
    lines.append(f"Analysis Date: {meta.get('analysis_date', 'N/A')}")

    # Warnings
    warnings = data.get('warnings', [])
    if warnings:
        lines.append(f"\nWarnings:")
        for w in warnings:
            lines.append(f"  - {w}")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def format_recent_output(data: dict) -> str:
    """Format recent transactions for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"RECENT INSIDER TRANSACTIONS: {data['symbol']}")
    lines.append(f"{'=' * 70}")

    summary = data.get('summary', {})
    lines.append(f"\nSummary ({data.get('period_days', 90)} days):")
    lines.append(f"  Total Transactions: {summary.get('total_count', 0)}")
    lines.append(f"  Purchase Value:     {format_currency(summary.get('purchase_value', 0))}")
    lines.append(f"  Sale Value:         {format_currency(summary.get('sale_value', 0))}")
    lines.append(f"  Net Value:          {format_currency(summary.get('net_value', 0))}")

    transactions = data.get('transactions', [])
    if transactions:
        lines.append(f"\n{'Date':<12} {'Type':<6} {'Insider':<25} {'Value':>12}")
        lines.append("-" * 60)
        for t in transactions[:20]:  # Limit to 20
            date = t.get('date', 'N/A')[:10] if t.get('date') else 'N/A'
            code = t.get('code', '?')
            name = (t.get('insider') or 'Unknown')[:24]
            value = t.get('value', 0)
            value_str = format_currency(abs(value))
            if code == 'S':
                value_str = f"-{value_str}"
            lines.append(f"{date:<12} {code:<6} {name:<25} {value_str:>12}")

        if len(transactions) > 20:
            lines.append(f"\n... and {len(transactions) - 20} more transactions")
    else:
        lines.append("\nNo transactions found in period.")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_clusters_output(data: dict) -> str:
    """Format cluster detection for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"CLUSTER DETECTION: {data['symbol']}")
    lines.append(f"{'=' * 60}")

    signal = data.get('cluster_signal', 'no_significant_clusters')
    lines.append(f"\nCluster Signal: {signal.upper()}")
    lines.append(f"Significant Clusters: {data.get('significant_clusters', 0)}")

    clusters = data.get('clusters', [])
    if clusters:
        for i, c in enumerate(clusters, 1):
            lines.append(f"\n--- Cluster {i} ---")
            lines.append(f"  Type:         {c.get('cluster_type', 'unknown')}")
            period = c.get('period', {})
            lines.append(f"  Period:       {period.get('start_date')} to {period.get('end_date')}")
            lines.append(f"  Days:         {period.get('days', 0)}")
            activity = c.get('activity', {})
            lines.append(f"  Insiders:     {activity.get('insider_count', 0)}")
            lines.append(f"  Transactions: {activity.get('transaction_count', 0)}")
            lines.append(f"  Total Value:  {format_currency(activity.get('total_value', 0))}")
            lines.append(f"  Significant:  {'Yes' if c.get('is_significant') else 'No'}")
            insiders = c.get('insiders', [])
            if insiders:
                lines.append(f"  Participants: {', '.join(insiders[:5])}")
                if len(insiders) > 5:
                    lines.append(f"                ... and {len(insiders) - 5} more")
    else:
        lines.append("\nNo significant clusters detected.")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def format_key_insiders_output(data: dict) -> str:
    """Format key insider summary for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"KEY INSIDER ACTIVITY: {data['symbol']}")
    lines.append(f"{'=' * 70}")

    lines.append(f"\nPeriod: {data.get('analysis_period_days', 180)} days")
    lines.append(f"Total Transactions: {data.get('total_transactions', 0)}")
    lines.append(f"Net Value: {format_currency(data.get('net_value', 0))}")
    lines.append(f"Direction: {data.get('direction', 'neutral').upper()}")

    insiders = data.get('key_insiders', [])
    if insiders:
        lines.append(f"\n{'Name':<30} {'Title':<20} {'Txns':>6} {'Net Value':>14}")
        lines.append("-" * 72)
        for ins in insiders:
            name = (ins.get('name') or 'Unknown')[:29]
            title = (ins.get('title') or 'N/A')[:19]
            txns = ins.get('transactions', 0)
            net = ins.get('net_value', 0)
            lines.append(f"{name:<30} {title:<20} {txns:>6} {format_currency(net):>14}")
    else:
        lines.append("\nNo key insider activity found.")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_fetch_output(data: dict) -> str:
    """Format fetch results for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 50}")
    lines.append(f"FETCH RESULTS: {data['symbol']}")
    lines.append(f"{'=' * 50}")
    lines.append(f"\nFilings Fetched: {data.get('filings_fetched', 0)}")
    lines.append(f"Filings Saved:   {data.get('filings_saved', 0)}")
    lines.append(f"\nMessage: {data.get('message', 'N/A')}")
    lines.append(f"\n{'=' * 50}")
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="SEC Form 4 Insider Trading Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL --sentiment          Get insider sentiment analysis
  %(prog)s MSFT --recent --days 60   Get recent transactions
  %(prog)s NVDA --clusters           Detect cluster activity
  %(prog)s TSLA --key-insiders       Get C-suite/director summary
  %(prog)s AAPL --fetch              Fetch fresh Form 4 data
        """
    )

    parser.add_argument(
        "symbol",
        help="Stock ticker symbol (e.g., AAPL, MSFT)"
    )

    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--sentiment",
        action="store_true",
        help="Get insider sentiment analysis"
    )
    action_group.add_argument(
        "--recent",
        action="store_true",
        help="Get recent insider transactions"
    )
    action_group.add_argument(
        "--clusters",
        action="store_true",
        help="Detect cluster buying/selling activity"
    )
    action_group.add_argument(
        "--key-insiders",
        action="store_true",
        help="Get key insider (C-suite, directors) summary"
    )
    action_group.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch fresh Form 4 filings from SEC EDGAR"
    )

    # Options
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Analysis period in days (default: 90)"
    )
    parser.add_argument(
        "--significant-only",
        action="store_true",
        help="For --recent: only show significant transactions"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text"
    )

    args = parser.parse_args()

    # Import and initialize tool
    from victor_invest.tools.insider_trading import InsiderTradingTool

    tool = InsiderTradingTool()
    await tool.initialize()

    # Determine action
    if args.sentiment:
        action = "sentiment"
    elif args.recent:
        action = "recent"
    elif args.clusters:
        action = "clusters"
    elif args.key_insiders:
        action = "key_insiders"
    elif args.fetch:
        action = "fetch"
    else:
        print("Error: No action specified", file=sys.stderr)
        sys.exit(1)

    # Execute
    result = await tool.execute(
        symbol=args.symbol,
        action=action,
        days=args.days,
        significant_only=args.significant_only
    )

    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        print(json.dumps(result.data, indent=2))
    else:
        if action == "sentiment":
            print(format_sentiment_output(result.data))
        elif action == "recent":
            print(format_recent_output(result.data))
        elif action == "clusters":
            print(format_clusters_output(result.data))
        elif action == "key_insiders":
            print(format_key_insiders_output(result.data))
        elif action == "fetch":
            print(format_fetch_output(result.data))


if __name__ == "__main__":
    asyncio.run(main())
