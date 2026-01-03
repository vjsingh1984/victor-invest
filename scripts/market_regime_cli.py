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

"""Market Regime CLI.

Command-line interface for comprehensive market regime analysis combining
yield curve, credit cycle, recession probability, and volatility indicators.

Usage:
    python scripts/market_regime_cli.py --summary
    python scripts/market_regime_cli.py --credit-cycle
    python scripts/market_regime_cli.py --recommendations
    python scripts/market_regime_cli.py --yield-curve
    python scripts/market_regime_cli.py --recession
    python scripts/market_regime_cli.py --volatility

Examples:
    # Get comprehensive market regime summary
    python scripts/market_regime_cli.py --summary

    # Get credit cycle analysis
    python scripts/market_regime_cli.py --credit-cycle

    # Get investment recommendations
    python scripts/market_regime_cli.py --recommendations

    # Get yield curve analysis
    python scripts/market_regime_cli.py --yield-curve
"""

import argparse
import asyncio
import json
import sys

# Add project root to path
sys.path.insert(0, "/Users/vijaysingh/code/victor-invest")


def format_summary_output(data: dict) -> str:
    """Format comprehensive market regime summary for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 75}")
    lines.append(f"MARKET REGIME SUMMARY")
    lines.append(f"{'=' * 75}")
    lines.append(f"\nDate: {data.get('date', 'N/A')}")

    # Overall Regime
    regime = data.get('overall_regime', {})
    lines.append(f"\nOVERALL REGIME")
    lines.append(f"  Credit Cycle Phase:   {regime.get('credit_cycle_phase', 'N/A').upper().replace('_', ' ')}")
    lines.append(f"  Yield Curve Shape:    {regime.get('yield_curve_shape', 'N/A').upper().replace('_', ' ')}")
    lines.append(f"  Investment Signal:    {regime.get('investment_signal', 'N/A').upper().replace('_', ' ')}")
    lines.append(f"  Confidence:           {regime.get('confidence', 0):.0%}")

    # Overall Signal
    signal = data.get('overall_signal', {})
    if signal:
        level = signal.get('level', 'unknown').upper().replace('_', ' ')
        lines.append(f"\nOVERALL SIGNAL: {level}")
        if signal.get('description'):
            lines.append(f"  {signal['description']}")

    # Key Indicators
    indicators = data.get('indicators', {})
    lines.append(f"\nKEY INDICATORS")
    lines.append(f"  BAA Credit Spread:    {indicators.get('baa_credit_spread_bps', 0):.0f} bps")
    lines.append(f"  VIX Level:            {indicators.get('vix_level', 0):.1f}")
    lines.append(f"  Fed Funds Rate:       {indicators.get('fed_funds_rate', 0):.2f}%")
    lines.append(f"  10Y-2Y Spread:        {indicators.get('yield_10y_2y_spread_bps', 0):.0f} bps")

    # Classifications
    classifications = data.get('classifications', {})
    lines.append(f"\nCLASSIFICATIONS")
    lines.append(f"  Credit Cycle:         {classifications.get('credit_cycle', 'N/A').upper().replace('_', ' ')}")
    lines.append(f"  Volatility Regime:    {classifications.get('volatility_regime', 'N/A').upper().replace('_', ' ')}")
    lines.append(f"  Fed Policy Stance:    {classifications.get('fed_policy_stance', 'N/A').upper().replace('_', ' ')}")
    lines.append(f"  Recession Prob:       {classifications.get('recession_probability', 'N/A').upper().replace('_', ' ')}")

    # Valuation Impacts
    val = data.get('valuation_impacts', {})
    lines.append(f"\nVALUATION IMPACTS")
    lines.append(f"  Risk-Free Rate:       {val.get('risk_free_rate', 0):.2f}%")
    lines.append(f"  WACC Spread Adj:      {val.get('wacc_spread_adjustment_bps', 0):+.0f} bps")
    equity_adj = val.get('equity_allocation_adjustment', 0)
    lines.append(f"  Equity Allocation:    {equity_adj * 100:+.0f}%")

    # Interpretation
    interpretation = data.get('interpretation', '')
    if interpretation:
        lines.append(f"\nINTERPRETATION:")
        _wrap_text(lines, interpretation, indent=2, width=70)

    # Sector Recommendations
    sectors = data.get('sector_recommendations', {})
    if sectors:
        lines.append(f"\nSECTOR RECOMMENDATIONS")
        overweight = sectors.get('overweight', [])
        underweight = sectors.get('underweight', [])
        if overweight:
            lines.append(f"  Overweight:  {', '.join(overweight)}")
        if underweight:
            lines.append(f"  Underweight: {', '.join(underweight)}")

    # Factors
    factors = data.get('factors', [])
    if factors:
        lines.append(f"\nKEY FACTORS:")
        for factor in factors[:5]:
            lines.append(f"  - {factor}")

    lines.append(f"\n{'=' * 75}")
    return "\n".join(lines)


def format_credit_cycle_output(data: dict) -> str:
    """Format credit cycle analysis for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"CREDIT CYCLE ANALYSIS")
    lines.append(f"{'=' * 70}")
    lines.append(f"\nDate: {data.get('date', 'N/A')}")

    # Phase with visual indicator
    phase = data.get('phase', 'unknown').upper().replace('_', ' ')
    confidence = data.get('confidence', 0)

    phase_indicator = _get_phase_indicator(phase)
    lines.append(f"\nCREDIT CYCLE PHASE: {phase}")
    lines.append(f"  {phase_indicator}")
    lines.append(f"  Confidence: {confidence:.0%}")

    # Indicators
    lines.append(f"\nINDICATORS")
    lines.append(f"  BAA Credit Spread:    {data.get('baa_spread_bps', 0):.0f} bps")
    lines.append(f"  VIX Level:            {data.get('vix_level', 0):.1f}")
    lines.append(f"  Fed Funds Rate:       {data.get('fed_funds_rate', 0):.2f}%")

    # Volatility Regime
    vol_regime = data.get('volatility_regime', 'normal').upper().replace('_', ' ')
    lines.append(f"\nVOLATILITY REGIME: {vol_regime}")

    # Fed Policy
    fed_stance = data.get('fed_policy_stance', 'neutral').upper().replace('_', ' ')
    lines.append(f"FED POLICY STANCE: {fed_stance}")

    # Recession Probability
    recession_prob = data.get('recession_probability', 'low').upper().replace('_', ' ')
    lines.append(f"RECESSION PROBABILITY: {recession_prob}")

    # Interpretation
    interpretation = data.get('interpretation', '')
    if interpretation:
        lines.append(f"\nINTERPRETATION:")
        _wrap_text(lines, interpretation, indent=2, width=66)

    # Sector Recommendations
    sectors = data.get('sector_recommendations', {})
    if sectors:
        lines.append(f"\nSECTOR RECOMMENDATIONS")
        overweight = sectors.get('overweight', [])
        underweight = sectors.get('underweight', [])
        if overweight:
            lines.append(f"  Overweight:  {', '.join(overweight)}")
        if underweight:
            lines.append(f"  Underweight: {', '.join(underweight)}")

    # Factors
    factors = data.get('factors', [])
    if factors:
        lines.append(f"\nCONTRIBUTING FACTORS:")
        for factor in factors:
            lines.append(f"  - {factor}")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_yield_curve_output(data: dict) -> str:
    """Format yield curve analysis for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"YIELD CURVE ANALYSIS")
    lines.append(f"{'=' * 70}")
    lines.append(f"\nDate: {data.get('date', 'N/A')}")

    # Shape
    shape = data.get('shape', 'normal').upper().replace('_', ' ')
    lines.append(f"\nYIELD CURVE SHAPE: {shape}")

    # Visual representation
    lines.append(f"\n  {_get_curve_visual(shape)}")

    # Spreads
    lines.append(f"\nYIELD SPREADS")
    lines.append(f"  10Y-2Y Spread:        {data.get('spread_10y_2y_bps', 0):.0f} bps")
    lines.append(f"  10Y-3M Spread:        {data.get('spread_10y_3m_bps', 0):.0f} bps")

    # Investment Signal
    signal = data.get('investment_signal', 'neutral').upper().replace('_', ' ')
    lines.append(f"\nINVESTMENT SIGNAL: {signal}")

    # Rates
    lines.append(f"\nKEY RATES")
    lines.append(f"  Risk-Free Rate:       {data.get('risk_free_rate', 0):.2f}%")
    lines.append(f"  WACC Spread Adj:      {data.get('wacc_spread_adjustment', 0):+.0f} bps")

    # Equity Adjustment
    equity_adj = data.get('equity_adjustment', 0)
    lines.append(f"  Equity Allocation:    {equity_adj * 100:+.0f}%")

    # Interpretation
    interpretation = data.get('interpretation', '')
    if interpretation:
        lines.append(f"\nINTERPRETATION:")
        _wrap_text(lines, interpretation, indent=2, width=66)

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_recession_output(data: dict) -> str:
    """Format recession assessment for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"RECESSION PROBABILITY ASSESSMENT")
    lines.append(f"{'=' * 70}")
    lines.append(f"\nDate: {data.get('date', 'N/A')}")

    # Economic Phase
    phase = data.get('phase', 'expansion').upper().replace('_', ' ')
    lines.append(f"\nECONOMIC PHASE: {phase}")

    # Probability bar
    prob_pct = data.get('probability_pct', 0)
    bar_len = 40
    filled = int(prob_pct / 100 * bar_len)
    bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'

    lines.append(f"\nRECESSION PROBABILITY: {prob_pct:.0f}%")
    lines.append(f"  {bar}")

    # Risk Level
    if prob_pct >= 75:
        lines.append("  [!!!] IMMINENT RECESSION RISK")
    elif prob_pct >= 50:
        lines.append("  [!!] HIGH RECESSION RISK")
    elif prob_pct >= 25:
        lines.append("  [!] ELEVATED RECESSION RISK")
    elif prob_pct >= 10:
        lines.append("  LOW RECESSION RISK")
    else:
        lines.append("  VERY LOW RECESSION RISK")

    # Indicators
    indicators = data.get('indicators', {})
    if indicators:
        lines.append(f"\nLEADING INDICATORS")
        for name, value in indicators.items():
            display_name = name.replace('_', ' ').title()
            if isinstance(value, float):
                lines.append(f"  {display_name:<25} {value:.2f}")
            else:
                lines.append(f"  {display_name:<25} {value}")

    # Interpretation
    interpretation = data.get('interpretation', '')
    if interpretation:
        lines.append(f"\nASSESSMENT:")
        _wrap_text(lines, interpretation, indent=2, width=66)

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_volatility_output(data: dict) -> str:
    """Format volatility regime analysis for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"VOLATILITY REGIME ANALYSIS")
    lines.append(f"{'=' * 70}")
    lines.append(f"\nDate: {data.get('date', 'N/A')}")

    # VIX Level
    vix = data.get('vix_level', 0)
    lines.append(f"\nVIX LEVEL: {vix:.1f}")

    # Visual bar
    bar_len = 40
    # Scale VIX (0-60 range typical)
    vix_scaled = min(vix / 60, 1.0)
    filled = int(vix_scaled * bar_len)
    bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
    lines.append(f"  {bar}")

    # Regime
    regime = data.get('volatility_regime', 'normal').upper().replace('_', ' ')
    lines.append(f"\nVOLATILITY REGIME: {regime}")

    # Interpretation
    interpretation = data.get('interpretation', '')
    if interpretation:
        lines.append(f"\nINTERPRETATION:")
        _wrap_text(lines, interpretation, indent=2, width=66)

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_recommendations_output(data: dict) -> str:
    """Format investment recommendations for terminal output."""
    lines = []
    lines.append(f"\n{'=' * 75}")
    lines.append(f"MARKET REGIME INVESTMENT RECOMMENDATIONS")
    lines.append(f"{'=' * 75}")
    lines.append(f"\nDate: {data.get('date', 'N/A')}")

    # Current Regime
    phase = data.get('credit_cycle_phase', 'unknown').upper().replace('_', ' ')
    signal = data.get('investment_signal', 'neutral').upper().replace('_', ' ')
    lines.append(f"\nCURRENT REGIME")
    lines.append(f"  Credit Cycle Phase:   {phase}")
    lines.append(f"  Investment Signal:    {signal}")

    # Allocation Guidance
    allocation = data.get('allocation_guidance', {})
    lines.append(f"\nALLOCATION GUIDANCE")
    lines.append(f"  Equity Adjustment:    {allocation.get('equity_adjustment', '0%')}")
    lines.append(f"  Risk Posture:         {allocation.get('risk_posture', 'Balanced')}")
    lines.append(f"  Duration Guidance:    {allocation.get('duration_guidance', 'Neutral')}")

    # Sector Recommendations
    sectors = data.get('sector_recommendations', {})
    if sectors:
        lines.append(f"\nSECTOR POSITIONING")
        overweight = sectors.get('overweight', [])
        underweight = sectors.get('underweight', [])
        neutral = sectors.get('neutral', [])
        if overweight:
            lines.append(f"  OVERWEIGHT:")
            for sector in overweight:
                lines.append(f"    + {sector}")
        if neutral:
            lines.append(f"  NEUTRAL:")
            for sector in neutral[:3]:
                lines.append(f"    = {sector}")
        if underweight:
            lines.append(f"  UNDERWEIGHT:")
            for sector in underweight:
                lines.append(f"    - {sector}")

    # Interpretation
    interpretation = data.get('interpretation', '')
    if interpretation:
        lines.append(f"\nMARKET ASSESSMENT:")
        _wrap_text(lines, interpretation, indent=2, width=70)

    lines.append(f"\n{'=' * 75}")
    return "\n".join(lines)


def _wrap_text(lines: list, text: str, indent: int = 2, width: int = 66) -> None:
    """Wrap text and append to lines with indentation."""
    prefix = " " * indent
    words = text.split()
    line = prefix
    for word in words:
        if len(line) + len(word) > width:
            lines.append(line)
            line = prefix
        line += word + " "
    if line.strip():
        lines.append(line)


def _get_phase_indicator(phase: str) -> str:
    """Get visual indicator for credit cycle phase."""
    indicators = {
        "EARLY EXPANSION": "[=====>........] Recovery phase",
        "MID CYCLE": "[=======>......] Healthy expansion",
        "LATE CYCLE": "[==========>...] Slowing growth",
        "CREDIT STRESS": "[============>.] Elevated risk",
        "CREDIT CRISIS": "[==============] Maximum defensive",
    }
    return indicators.get(phase, "[..............]")


def _get_curve_visual(shape: str) -> str:
    """Get visual representation of yield curve shape."""
    visuals = {
        "STEEP": "     /--     Steep (rates rising with maturity)",
        "NORMAL": "    /        Normal (gradual rise)",
        "FLAT": "   ___       Flat (similar rates)",
        "INVERTED": "   \\         Inverted (short > long)",
        "DEEPLY INVERTED": "    \\\\__     Deeply Inverted (recession signal)",
    }
    return visuals.get(shape, "   ???       Unknown shape")


async def main():
    parser = argparse.ArgumentParser(
        description="Market Regime Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --summary                Get comprehensive market regime summary
  %(prog)s --credit-cycle           Get credit cycle analysis
  %(prog)s --yield-curve            Get yield curve analysis
  %(prog)s --recession              Get recession probability assessment
  %(prog)s --volatility             Get volatility regime analysis
  %(prog)s --recommendations        Get investment recommendations
        """
    )

    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--summary",
        action="store_true",
        help="Get comprehensive market regime summary"
    )
    action_group.add_argument(
        "--credit-cycle",
        action="store_true",
        help="Get credit cycle analysis"
    )
    action_group.add_argument(
        "--yield-curve",
        action="store_true",
        help="Get yield curve analysis"
    )
    action_group.add_argument(
        "--recession",
        action="store_true",
        help="Get recession probability assessment"
    )
    action_group.add_argument(
        "--volatility",
        action="store_true",
        help="Get volatility regime analysis"
    )
    action_group.add_argument(
        "--recommendations",
        action="store_true",
        help="Get investment recommendations based on regime"
    )

    # Options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text"
    )

    args = parser.parse_args()

    # Import and initialize tool
    from victor_invest.tools.market_regime import MarketRegimeTool

    tool = MarketRegimeTool()
    await tool.initialize()

    # Determine action and execute
    if args.summary:
        result = await tool.execute(action="summary")
        formatter = format_summary_output
    elif args.credit_cycle:
        result = await tool.execute(action="credit_cycle")
        formatter = format_credit_cycle_output
    elif args.yield_curve:
        result = await tool.execute(action="yield_curve")
        formatter = format_yield_curve_output
    elif args.recession:
        result = await tool.execute(action="recession")
        formatter = format_recession_output
    elif args.volatility:
        result = await tool.execute(action="volatility")
        formatter = format_volatility_output
    elif args.recommendations:
        result = await tool.execute(action="recommendations")
        formatter = format_recommendations_output
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
