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

"""Valuation Signals CLI.

Command-line interface for integrated valuation signal analysis combining
credit risk, insider sentiment, short interest, and market regime data.

Usage:
    python scripts/valuation_signals_cli.py --integrate AAPL --base-fv 190 --price 185
    python scripts/valuation_signals_cli.py --credit-risk AAPL
    python scripts/valuation_signals_cli.py --insider AAPL
    python scripts/valuation_signals_cli.py --short-interest AAPL
    python scripts/valuation_signals_cli.py --market-regime

Examples:
    # Full signal integration
    python scripts/valuation_signals_cli.py --integrate AAPL --base-fv 190.0 --price 185.0

    # Credit risk analysis only
    python scripts/valuation_signals_cli.py --credit-risk AAPL

    # With manual credit risk inputs
    python scripts/valuation_signals_cli.py --credit-risk AAPL \\
        --altman-z 2.5 --beneish-m -2.1 --piotroski-f 7
"""

import argparse
import asyncio
import json
import sys

# Add project root to path
sys.path.insert(0, "/Users/vijaysingh/code/victor-invest")


def format_integrated_output(data: dict) -> str:
    """Format integrated valuation signals for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"INTEGRATED VALUATION SIGNALS - {data.get('symbol', 'N/A')}")
    lines.append(f"{'=' * 80}")

    # Fair Value Summary
    base_fv = data.get('base_fair_value', 0)
    adj_fv = data.get('adjusted_fair_value', 0)
    price = data.get('current_price', 0)
    upside = data.get('upside_pct', 0)
    adjustment = data.get('total_adjustment_pct', 0)

    lines.append(f"\nFAIR VALUE SUMMARY")
    lines.append(f"  Base Fair Value:      ${base_fv:.2f}")
    lines.append(f"  Signal Adjustment:    {adjustment:+.1f}%")
    lines.append(f"  Adjusted Fair Value:  ${adj_fv:.2f}")
    lines.append(f"  Current Price:        ${price:.2f}")
    lines.append(f"  Implied Upside:       {upside:+.1f}%")

    # Adjustment bar
    bar_len = 40
    adj_normalized = max(-50, min(50, adjustment)) / 50  # Normalize to -1 to 1
    center = bar_len // 2
    if adj_normalized >= 0:
        filled = int(adj_normalized * center)
        bar = '[' + '-' * center + '#' * filled + '-' * (center - filled) + ']'
    else:
        filled = int(-adj_normalized * center)
        bar = '[' + '-' * (center - filled) + '#' * filled + '-' * center + ']'
    lines.append(f"  {bar}")
    lines.append(f"  {'Discount':^20}|{'Premium':^20}")

    # Credit Risk
    credit = data.get('credit_risk')
    if credit:
        lines.append(f"\nCREDIT RISK SIGNAL")
        lines.append(f"  Distress Tier:    {credit.get('distress_tier', 'N/A').upper().replace('_', ' ')}")
        lines.append(f"  Discount Applied: {credit.get('discount_pct', 0)*100:.0f}%")
        if credit.get('altman_zscore'):
            lines.append(f"  Altman Z-Score:   {credit['altman_zscore']:.2f} ({credit.get('altman_zone', 'N/A')})")
        if credit.get('beneish_mscore'):
            flag = " [MANIPULATION RISK]" if credit.get('manipulation_flag') else ""
            lines.append(f"  Beneish M-Score:  {credit['beneish_mscore']:.2f}{flag}")
        if credit.get('piotroski_fscore') is not None:
            lines.append(f"  Piotroski F-Score: {credit['piotroski_fscore']}/9 ({credit.get('piotroski_grade', 'N/A')})")
        if credit.get('factors'):
            lines.append(f"  Factors:")
            for factor in credit['factors'][:3]:
                lines.append(f"    - {factor}")

    # Insider Sentiment
    insider = data.get('insider_sentiment')
    if insider:
        lines.append(f"\nINSIDER SENTIMENT SIGNAL")
        signal = insider.get('signal', 'neutral').upper().replace('_', ' ')
        conf_adj = insider.get('confidence_adjustment', 0)
        lines.append(f"  Signal:              {signal}")
        lines.append(f"  Confidence Adj:      {conf_adj*100:+.0f}%")
        if insider.get('buy_sell_ratio'):
            lines.append(f"  Buy/Sell Ratio:      {insider['buy_sell_ratio']:.2f}")
        if insider.get('cluster_detected'):
            lines.append(f"  Cluster Detected:    YES [AMPLIFIED SIGNAL]")
        if insider.get('interpretation'):
            lines.append(f"  Interpretation:      {insider['interpretation']}")

    # Short Interest
    short = data.get('short_interest')
    if short:
        lines.append(f"\nSHORT INTEREST SIGNAL")
        signal = short.get('signal', 'normal').upper().replace('_', ' ')
        lines.append(f"  Signal:              {signal}")
        if short.get('short_percent_float'):
            lines.append(f"  Short % of Float:    {short['short_percent_float']:.1f}%")
        if short.get('days_to_cover'):
            lines.append(f"  Days to Cover:       {short['days_to_cover']:.1f}")
        if short.get('squeeze_score'):
            lines.append(f"  Squeeze Score:       {short['squeeze_score']:.0f}/100")
        if short.get('is_contrarian_signal'):
            lines.append(f"  Contrarian Signal:   YES [BULLISH]")
        if short.get('warning_flag'):
            lines.append(f"  Warning:             {short.get('interpretation', 'Elevated short interest')}")

    # Market Regime
    regime = data.get('market_regime')
    if regime:
        lines.append(f"\nMARKET REGIME ADJUSTMENT")
        phase = regime.get('credit_cycle_phase', 'mid_cycle').upper().replace('_', ' ')
        vol = regime.get('volatility_regime', 'normal').upper()
        rec = regime.get('recession_probability', 'low').upper()
        lines.append(f"  Credit Cycle Phase:  {phase}")
        lines.append(f"  Volatility Regime:   {vol}")
        lines.append(f"  Recession Prob:      {rec}")
        wacc = regime.get('wacc_spread_adjustment_bps', 0)
        lines.append(f"  WACC Adjustment:     {wacc:+d} bps")
        val_factor = regime.get('valuation_adjustment_factor', 1.0)
        lines.append(f"  Valuation Factor:    {val_factor:.2f}x")
        if regime.get('factors'):
            lines.append(f"  Factors:")
            for factor in regime['factors'][:3]:
                lines.append(f"    - {factor}")

    # Warnings
    warnings = data.get('warnings', [])
    if warnings:
        lines.append(f"\nWARNINGS")
        for warning in warnings:
            lines.append(f"  [!] {warning}")

    lines.append(f"\n{'=' * 80}")
    return "\n".join(lines)


def format_credit_risk_output(data: dict) -> str:
    """Format credit risk signal for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"CREDIT RISK SIGNAL")
    lines.append(f"{'=' * 70}")

    tier = data.get('distress_tier', 'healthy').upper().replace('_', ' ')
    discount = data.get('discount_pct', 0) * 100

    lines.append(f"\nDISTRESS TIER: {tier}")
    lines.append(f"VALUATION DISCOUNT: {discount:.0f}%")

    # Visual tier indicator
    tiers = ["HEALTHY", "WATCH", "CONCERN", "DISTRESSED", "SEVERE DISTRESS"]
    tier_index = tiers.index(tier) if tier in tiers else 0
    indicator = '[' + '=' * (tier_index + 1) + '>' + '.' * (4 - tier_index) + ']'
    lines.append(f"\n  {indicator}")
    lines.append(f"  Healthy → → → → Severe Distress")

    # Scores
    lines.append(f"\nSCORES")
    if data.get('altman_zscore'):
        zone = data.get('altman_zone', 'unknown').upper()
        lines.append(f"  Altman Z-Score:    {data['altman_zscore']:.2f} [{zone}]")
        lines.append(f"                     >2.99=Safe, 1.81-2.99=Grey, <1.81=Distress")
    if data.get('beneish_mscore'):
        flag = " [MANIPULATION RISK]" if data.get('manipulation_flag') else " [OK]"
        lines.append(f"  Beneish M-Score:   {data['beneish_mscore']:.2f}{flag}")
        lines.append(f"                     >-1.78=Manipulation likely")
    if data.get('piotroski_fscore') is not None:
        grade = data.get('piotroski_grade', 'unknown').upper()
        lines.append(f"  Piotroski F-Score: {data['piotroski_fscore']}/9 [{grade}]")
        lines.append(f"                     8-9=Strong, 5-7=Moderate, 0-4=Weak")

    # Factors
    factors = data.get('factors', [])
    if factors:
        lines.append(f"\nFACTORS")
        for factor in factors:
            lines.append(f"  - {factor}")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_insider_output(data: dict) -> str:
    """Format insider sentiment for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"INSIDER SENTIMENT SIGNAL")
    lines.append(f"{'=' * 70}")

    signal = data.get('signal', 'neutral').upper().replace('_', ' ')
    conf_adj = data.get('confidence_adjustment', 0) * 100

    lines.append(f"\nSIGNAL: {signal}")
    lines.append(f"CONFIDENCE ADJUSTMENT: {conf_adj:+.0f}%")

    # Visual signal indicator
    signals = ["STRONG SELL", "SELL", "NEUTRAL", "BUY", "STRONG BUY"]
    try:
        sig_index = signals.index(signal)
    except ValueError:
        sig_index = 2  # Default to NEUTRAL
    indicator = '[' + '-' * sig_index + '#' + '-' * (4 - sig_index) + ']'
    lines.append(f"\n  {indicator}")
    lines.append(f"  Sell ← → → → → Buy")

    # Metrics
    lines.append(f"\nMETRICS")
    if data.get('buy_sell_ratio'):
        lines.append(f"  Buy/Sell Ratio:      {data['buy_sell_ratio']:.2f}")
    if data.get('net_shares_change'):
        lines.append(f"  Net Shares Change:   {data['net_shares_change']:,}")
    if data.get('cluster_detected'):
        lines.append(f"  Cluster Detected:    YES [SIGNAL AMPLIFIED]")

    # Interpretation
    if data.get('interpretation'):
        lines.append(f"\nINTERPRETATION:")
        lines.append(f"  {data['interpretation']}")

    # Factors
    factors = data.get('factors', [])
    if factors:
        lines.append(f"\nFACTORS")
        for factor in factors:
            lines.append(f"  - {factor}")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_short_interest_output(data: dict) -> str:
    """Format short interest signal for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"SHORT INTEREST SIGNAL")
    lines.append(f"{'=' * 70}")

    signal = data.get('signal', 'normal').upper().replace('_', ' ')

    lines.append(f"\nSIGNAL: {signal}")

    if data.get('is_contrarian_signal'):
        lines.append(f"CONTRARIAN: BULLISH (Potential squeeze opportunity)")
    if data.get('warning_flag'):
        lines.append(f"WARNING: Elevated short interest")

    # Metrics
    lines.append(f"\nMETRICS")
    if data.get('short_percent_float'):
        pct = data['short_percent_float']
        bar_len = 40
        filled = int(min(pct / 50, 1.0) * bar_len)
        bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
        lines.append(f"  Short % of Float:    {pct:.1f}%")
        lines.append(f"  {bar}")
        lines.append(f"  0%        |        25%        |        50%+")
    if data.get('days_to_cover'):
        lines.append(f"  Days to Cover:       {data['days_to_cover']:.1f}")
    if data.get('squeeze_score'):
        score = data['squeeze_score']
        lines.append(f"  Squeeze Score:       {score:.0f}/100")
        if score >= 70:
            lines.append(f"                       [HIGH SQUEEZE RISK]")

    # Interpretation
    if data.get('interpretation'):
        lines.append(f"\nINTERPRETATION:")
        lines.append(f"  {data['interpretation']}")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_market_regime_output(data: dict) -> str:
    """Format market regime adjustment for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"MARKET REGIME ADJUSTMENT")
    lines.append(f"{'=' * 70}")

    phase = data.get('credit_cycle_phase', 'mid_cycle').upper().replace('_', ' ')
    vol = data.get('volatility_regime', 'normal').upper()
    rec = data.get('recession_probability', 'low').upper()
    fed = data.get('fed_policy_stance', 'neutral').upper()

    lines.append(f"\nCREDIT CYCLE PHASE: {phase}")

    # Phase indicator
    phases = ["EARLY EXPANSION", "MID CYCLE", "LATE CYCLE", "CREDIT STRESS", "CREDIT CRISIS"]
    try:
        phase_index = phases.index(phase)
    except ValueError:
        phase_index = 1
    indicator = '[' + '=' * (phase_index + 1) + '>' + '.' * (4 - phase_index) + ']'
    lines.append(f"  {indicator}")
    lines.append(f"  Expansion → → → → Crisis")

    lines.append(f"\nREGIME INDICATORS")
    lines.append(f"  Volatility:          {vol}")
    lines.append(f"  Recession Prob:      {rec}")
    lines.append(f"  Fed Policy:          {fed}")
    lines.append(f"  Risk-Free Rate:      {data.get('risk_free_rate', 0.04)*100:.2f}%")

    lines.append(f"\nVALUATION ADJUSTMENTS")
    wacc = data.get('wacc_spread_adjustment_bps', 0)
    lines.append(f"  WACC Spread Adj:     {wacc:+d} bps")
    eq_adj = data.get('equity_allocation_adjustment', 0)
    lines.append(f"  Equity Allocation:   {eq_adj*100:+.0f}%")
    val_factor = data.get('valuation_adjustment_factor', 1.0)
    lines.append(f"  Valuation Factor:    {val_factor:.2f}x")

    # Interpretation
    if data.get('interpretation'):
        lines.append(f"\nINTERPRETATION:")
        lines.append(f"  {data['interpretation']}")

    # Factors
    factors = data.get('factors', [])
    if factors:
        lines.append(f"\nFACTORS")
        for factor in factors:
            lines.append(f"  - {factor}")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="Valuation Signals CLI - Integrated signal analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --integrate AAPL --base-fv 190 --price 185   Full signal integration
  %(prog)s --credit-risk AAPL                           Credit risk signal only
  %(prog)s --insider AAPL                               Insider sentiment only
  %(prog)s --short-interest GME                         Short interest signal
  %(prog)s --market-regime                              Market regime adjustment
        """
    )

    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--integrate",
        metavar="SYMBOL",
        help="Full signal integration for symbol"
    )
    action_group.add_argument(
        "--credit-risk",
        metavar="SYMBOL",
        help="Credit risk signal for symbol"
    )
    action_group.add_argument(
        "--insider",
        metavar="SYMBOL",
        help="Insider sentiment signal for symbol"
    )
    action_group.add_argument(
        "--short-interest",
        metavar="SYMBOL",
        help="Short interest signal for symbol"
    )
    action_group.add_argument(
        "--market-regime",
        action="store_true",
        help="Market regime adjustment (no symbol needed)"
    )

    # Parameters for integration
    parser.add_argument(
        "--base-fv",
        type=float,
        help="Base fair value from valuation models (required for --integrate)"
    )
    parser.add_argument(
        "--price",
        type=float,
        help="Current stock price (required for --integrate)"
    )

    # Manual credit risk inputs
    parser.add_argument("--altman-z", type=float, help="Altman Z-Score")
    parser.add_argument("--beneish-m", type=float, help="Beneish M-Score")
    parser.add_argument("--piotroski-f", type=int, help="Piotroski F-Score")

    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text"
    )

    args = parser.parse_args()

    # Import and initialize tool
    from victor_invest.tools.valuation_signals import ValuationSignalsTool

    tool = ValuationSignalsTool()
    await tool.initialize()

    # Execute based on action
    if args.integrate:
        if not args.base_fv or not args.price:
            print("Error: --base-fv and --price are required for --integrate", file=sys.stderr)
            sys.exit(1)
        result = await tool.execute(
            action="integrate",
            symbol=args.integrate,
            base_fair_value=args.base_fv,
            current_price=args.price,
        )
        formatter = format_integrated_output

    elif args.credit_risk:
        kwargs = {}
        if args.altman_z is not None:
            kwargs["altman_zscore"] = args.altman_z
        if args.beneish_m is not None:
            kwargs["beneish_mscore"] = args.beneish_m
        if args.piotroski_f is not None:
            kwargs["piotroski_fscore"] = args.piotroski_f
        result = await tool.execute(
            action="credit_risk",
            symbol=args.credit_risk,
            **kwargs
        )
        formatter = format_credit_risk_output

    elif args.insider:
        result = await tool.execute(action="insider", symbol=args.insider)
        formatter = format_insider_output

    elif args.short_interest:
        result = await tool.execute(action="short_interest", symbol=args.short_interest)
        formatter = format_short_interest_output

    elif args.market_regime:
        result = await tool.execute(action="market_regime")
        formatter = format_market_regime_output

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
