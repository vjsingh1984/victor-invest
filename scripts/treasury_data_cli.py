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

"""Treasury Data CLI.

Command-line interface for Treasury yield curve and market regime data.

Usage:
    python scripts/treasury_data_cli.py --curve
    python scripts/treasury_data_cli.py --spread
    python scripts/treasury_data_cli.py --regime
    python scripts/treasury_data_cli.py --recession
    python scripts/treasury_data_cli.py --summary
    python scripts/treasury_data_cli.py --history --maturity 10y --days 30

Examples:
    # Get current yield curve
    python scripts/treasury_data_cli.py --curve

    # Get yield spread analysis (10Y-2Y, 10Y-3M)
    python scripts/treasury_data_cli.py --spread

    # Get market regime from yield curve
    python scripts/treasury_data_cli.py --regime

    # Get recession probability
    python scripts/treasury_data_cli.py --recession

    # Get comprehensive summary
    python scripts/treasury_data_cli.py --summary
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, "/Users/vijaysingh/code/victor-invest")


def format_yield_curve_output(data: dict) -> str:
    """Format yield curve data for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"TREASURY YIELD CURVE")
    lines.append(f"{'=' * 60}")
    lines.append(f"\nDate: {data.get('date', 'N/A')}")

    yields = data.get('yields', {})
    lines.append(f"\nYields by Maturity:")
    lines.append(f"  {'Maturity':<12} {'Yield':>10}")
    lines.append(f"  {'-' * 24}")

    maturities = [
        ('1m', '1 Month'),
        ('3m', '3 Month'),
        ('6m', '6 Month'),
        ('1y', '1 Year'),
        ('2y', '2 Year'),
        ('3y', '3 Year'),
        ('5y', '5 Year'),
        ('7y', '7 Year'),
        ('10y', '10 Year'),
        ('20y', '20 Year'),
        ('30y', '30 Year'),
    ]

    for key, label in maturities:
        value = yields.get(key)
        if value is not None:
            lines.append(f"  {label:<12} {value:>9.3f}%")

    # Spreads
    spreads = data.get('spreads', {})
    lines.append(f"\nKey Spreads:")
    spread_10y_2y = spreads.get('10y_2y_bps')
    spread_10y_3m = spreads.get('10y_3m_bps')

    if spread_10y_2y is not None:
        status = "INVERTED" if spread_10y_2y < 0 else "Normal"
        lines.append(f"  10Y - 2Y: {spread_10y_2y:+.1f} bps ({status})")
    if spread_10y_3m is not None:
        status = "INVERTED" if spread_10y_3m < 0 else "Normal"
        lines.append(f"  10Y - 3M: {spread_10y_3m:+.1f} bps ({status})")

    # Inversion status
    inversion = data.get('inversion', {})
    if inversion.get('is_deeply_inverted'):
        lines.append(f"\n  [!] DEEPLY INVERTED - Recession warning")
    elif inversion.get('is_inverted'):
        lines.append(f"\n  [!] INVERTED - Historical recession indicator")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def format_spread_output(data: dict) -> str:
    """Format spread analysis for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"YIELD SPREAD ANALYSIS")
    lines.append(f"{'=' * 60}")
    lines.append(f"\nDate: {data.get('date', 'N/A')}")

    spreads = data.get('spreads', {})
    lines.append(f"\nSpreads (in basis points):")
    lines.append(f"  10Y - 2Y:  {spreads.get('10y_2y_bps', 'N/A'):+.1f} bps")
    lines.append(f"  10Y - 3M:  {spreads.get('10y_3m_bps', 'N/A'):+.1f} bps")

    inversion = data.get('inversion', {})
    lines.append(f"\nInversion Status:")
    lines.append(f"  Is Inverted:        {'Yes' if inversion.get('is_inverted') else 'No'}")
    lines.append(f"  Deeply Inverted:    {'Yes' if inversion.get('is_deeply_inverted') else 'No'}")
    lines.append(f"  Days Inverted:      {inversion.get('days_inverted', 0)}")

    lines.append(f"\nCurve Shape: {data.get('curve_shape', 'unknown').upper()}")
    lines.append(f"Investment Signal: {data.get('investment_signal', 'unknown').upper()}")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def format_regime_output(data: dict) -> str:
    """Format market regime for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"YIELD CURVE MARKET REGIME")
    lines.append(f"{'=' * 60}")
    lines.append(f"\nDate: {data.get('date', 'N/A')}")

    lines.append(f"\nCurve Shape: {data.get('shape', 'unknown').upper()}")

    spreads = data.get('spreads', {})
    lines.append(f"\nSpreads:")
    lines.append(f"  10Y - 2Y: {spreads.get('10y_2y_bps', 'N/A'):+.1f} bps")
    lines.append(f"  10Y - 3M: {spreads.get('10y_3m_bps', 'N/A'):+.1f} bps")

    signals = data.get('signals', {})
    lines.append(f"\nInvestment Signals:")
    lines.append(f"  Signal:         {signals.get('investment_signal', 'unknown').upper()}")
    lines.append(f"  Risk-Free Rate: {signals.get('risk_free_rate', 'N/A')}%")

    inversion = data.get('inversion', {})
    if inversion.get('is_inverted'):
        lines.append(f"\n  [!] Yield curve inverted for {inversion.get('days_inverted', 0)} days")

    interpretation = data.get('interpretation', '')
    if interpretation:
        lines.append(f"\nInterpretation:")
        # Wrap long text
        words = interpretation.split()
        line = "  "
        for word in words:
            if len(line) + len(word) > 58:
                lines.append(line)
                line = "  "
            line += word + " "
        if line.strip():
            lines.append(line)

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def format_recession_output(data: dict) -> str:
    """Format recession assessment for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 65}")
    lines.append(f"RECESSION PROBABILITY ASSESSMENT")
    lines.append(f"{'=' * 65}")
    lines.append(f"\nDate: {data.get('date', 'N/A')}")

    recession = data.get('recession', {})
    prob = recession.get('probability_pct', 0)

    # Probability bar
    bar_len = 40
    filled = int(prob / 100 * bar_len)
    bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
    lines.append(f"\nRecession Probability: {prob:.1f}%")
    lines.append(f"  {bar}")

    lines.append(f"\nEconomic Phase:       {recession.get('phase', 'unknown').upper()}")
    lines.append(f"Investment Posture:   {recession.get('investment_posture', 'unknown').upper()}")

    yield_curve = data.get('yield_curve', {})
    lines.append(f"\nYield Curve Indicators:")
    lines.append(f"  Inverted:           {'Yes' if yield_curve.get('is_inverted') else 'No'}")
    lines.append(f"  Days Inverted:      {yield_curve.get('inversion_days', 0)}")

    supply_chain = data.get('supply_chain', {})
    if supply_chain.get('gscpi_value') is not None:
        lines.append(f"\nSupply Chain (GSCPI):")
        lines.append(f"  Value:              {supply_chain.get('gscpi_value', 'N/A'):.2f} std dev")
        lines.append(f"  Stressed:           {'Yes' if supply_chain.get('is_stressed') else 'No'}")

    interpretation = data.get('interpretation', '')
    if interpretation:
        lines.append(f"\nAssessment:")
        words = interpretation.split()
        line = "  "
        for word in words:
            if len(line) + len(word) > 62:
                lines.append(line)
                line = "  "
            line += word + " "
        if line.strip():
            lines.append(line)

    lines.append(f"\nConfidence: {data.get('confidence', 0) * 100:.0f}%")

    lines.append(f"\n{'=' * 65}")
    return "\n".join(lines)


def format_summary_output(data: dict) -> str:
    """Format comprehensive summary for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"MARKET REGIME SUMMARY")
    lines.append(f"{'=' * 70}")
    lines.append(f"\nDate: {data.get('date', 'N/A')}")

    lines.append(f"\nECONOMIC ASSESSMENT")
    lines.append(f"  Phase:              {data.get('economic_phase', 'unknown').upper()}")
    lines.append(f"  Investment Posture: {data.get('investment_posture', 'unknown').upper()}")

    risk = data.get('risk_metrics', {})
    lines.append(f"\nRISK METRICS")
    lines.append(f"  Recession Prob:     {risk.get('recession_probability', 0):.1f}%")
    lines.append(f"  Curve Inverted:     {'Yes' if risk.get('yield_curve_inverted') else 'No'}")
    lines.append(f"  Supply Chain Stress: {'Yes' if risk.get('supply_chain_stress') else 'No'}")

    guidance = data.get('allocation_guidance', {})
    equity_range = guidance.get('equity_range', (0, 0))
    lines.append(f"\nALLOCATION GUIDANCE")
    lines.append(f"  Equity Range:       {equity_range[0]}% - {equity_range[1]}%")

    sectors = guidance.get('sector_recommendations', {})
    if sectors:
        lines.append(f"\n  Sector Recommendations:")
        for category, recommendation in sectors.items():
            lines.append(f"    {category.title()}: {recommendation}")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_history_output(data: dict) -> str:
    """Format historical data for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 50}")
    lines.append(f"HISTORICAL YIELDS: {data.get('maturity', '').upper()}")
    lines.append(f"{'=' * 50}")

    summary = data.get('summary', {})
    lines.append(f"\nPeriod: {data.get('period_days', 0)} days")
    lines.append(f"Data Points: {data.get('data_points', 0)}")

    lines.append(f"\nSummary:")
    lines.append(f"  Current: {summary.get('current', 'N/A'):.3f}%" if summary.get('current') else "  Current: N/A")
    lines.append(f"  Average: {summary.get('average', 'N/A'):.3f}%" if summary.get('average') else "  Average: N/A")
    lines.append(f"  Min:     {summary.get('min', 'N/A'):.3f}%" if summary.get('min') else "  Min: N/A")
    lines.append(f"  Max:     {summary.get('max', 'N/A'):.3f}%" if summary.get('max') else "  Max: N/A")

    history = data.get('history', [])
    if history:
        lines.append(f"\nRecent Data (last 10):")
        lines.append(f"  {'Date':<12} {'Yield':>10}")
        lines.append(f"  {'-' * 24}")
        for entry in history[:10]:
            lines.append(f"  {entry.get('date', 'N/A'):<12} {entry.get('yield', 0):>9.3f}%")

    lines.append(f"\n{'=' * 50}")
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="Treasury Yield Curve and Market Regime CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --curve          Get current yield curve
  %(prog)s --spread         Get yield spread analysis
  %(prog)s --regime         Get market regime from yield curve
  %(prog)s --recession      Get recession probability
  %(prog)s --summary        Get comprehensive summary
  %(prog)s --history --maturity 10y --days 30
        """
    )

    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--curve",
        action="store_true",
        help="Get current yield curve"
    )
    action_group.add_argument(
        "--spread",
        action="store_true",
        help="Get yield spread analysis"
    )
    action_group.add_argument(
        "--regime",
        action="store_true",
        help="Get market regime from yield curve"
    )
    action_group.add_argument(
        "--recession",
        action="store_true",
        help="Get recession probability assessment"
    )
    action_group.add_argument(
        "--summary",
        action="store_true",
        help="Get comprehensive market regime summary"
    )
    action_group.add_argument(
        "--history",
        action="store_true",
        help="Get historical yield data"
    )

    # Options
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days for historical data (default: 365)"
    )
    parser.add_argument(
        "--maturity",
        type=str,
        default="10y",
        help="Maturity for historical data (default: 10y)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text"
    )

    args = parser.parse_args()

    # Import and initialize tool
    from victor_invest.tools.treasury_data import TreasuryDataTool

    tool = TreasuryDataTool()
    await tool.initialize()

    # Determine action
    if args.curve:
        action = "curve"
    elif args.spread:
        action = "spread"
    elif args.regime:
        action = "regime"
    elif args.recession:
        action = "recession"
    elif args.summary:
        action = "summary"
    elif args.history:
        action = "history"
    else:
        print("Error: No action specified", file=sys.stderr)
        sys.exit(1)

    # Execute
    result = await tool.execute(
        action=action,
        days=args.days,
        maturity=args.maturity
    )

    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        print(json.dumps(result.data, indent=2))
    else:
        if action == "curve":
            print(format_yield_curve_output(result.data))
        elif action == "spread":
            print(format_spread_output(result.data))
        elif action == "regime":
            print(format_regime_output(result.data))
        elif action == "recession":
            print(format_recession_output(result.data))
        elif action == "summary":
            print(format_summary_output(result.data))
        elif action == "history":
            print(format_history_output(result.data))


if __name__ == "__main__":
    asyncio.run(main())
