"""
Valuation Table Formatter

Provides formatted ASCII table output for valuation model calculations.
Consolidates verbose multi-line logs into single, readable table records.

Author: InvestiGator Team
Date: 2025-11-07
"""

from typing import Any, Dict, List, Optional


class ValuationTableFormatter:
    """Formats valuation model calculations as ASCII tables for logging."""

    @staticmethod
    def format_dcf_table(
        symbol: str,
        inputs: Dict[str, Any],
        wacc_breakdown: Dict[str, float],
        projections: List[Dict[str, float]],
        terminal: Dict[str, float],
        valuation: Dict[str, float],
    ) -> str:
        """
        Format DCF valuation as comprehensive table.

        Args:
            symbol: Stock symbol
            inputs: TTM FCF, revenue growth, FCF margin, Rule of 40, etc.
            wacc_breakdown: Risk-free rate, beta, ERP, cost of equity, cost of debt, weights
            projections: List of {year, fcf, pv_fcf} dicts
            terminal: Terminal growth, terminal value, pv_terminal
            valuation: Enterprise value, equity value, fair value per share, current price, upside

        Returns:
            Formatted ASCII table string
        """
        lines = []
        lines.append(f"\n{'='*100}")
        lines.append(f"  DCF VALUATION - {symbol}")
        lines.append(f"{'='*100}")

        # Section 1: Inputs
        lines.append(f"\n{'─'*100}")
        lines.append("  📊 INPUTS")
        lines.append(f"{'─'*100}")
        lines.append(f"  TTM Free Cash Flow         : ${inputs.get('ttm_fcf', 0)/1e9:>8.2f}B")
        lines.append(f"  TTM Revenue                : ${inputs.get('ttm_revenue', 0)/1e9:>8.2f}B")
        lines.append(f"  FCF Margin                 : {inputs.get('fcf_margin', 0):>8.1f}%")
        lines.append(f"  Historical FCF Growth      : {inputs.get('fcf_growth', 0):>8.1f}%  (geometric mean)")
        lines.append(f"  Revenue Growth             : {inputs.get('revenue_growth', 0):>8.1f}%  (geometric mean)")
        lines.append(
            f"  Rule of 40 Score           : {inputs.get('rule_of_40', 0):>8.1f}%  ({inputs.get('rule_of_40_label', 'N/A')})"
        )
        lines.append(f"  Projection Years           : {inputs.get('projection_years', 5):>8.0f}")

        # Section 2: WACC Breakdown
        lines.append(f"\n{'─'*100}")
        lines.append("  💰 WACC CALCULATION")
        lines.append(f"{'─'*100}")
        lines.append(f"  Risk-Free Rate (10Y)       : {wacc_breakdown.get('rf_rate', 0):>8.2f}%")
        lines.append(f"  Beta                       : {wacc_breakdown.get('beta', 0):>8.2f}")
        lines.append(f"  Equity Risk Premium        : {wacc_breakdown.get('erp', 0):>8.2f}%")
        lines.append(f"  Cost of Equity             : {wacc_breakdown.get('cost_of_equity', 0):>8.2f}%  (Rf + β × ERP)")
        lines.append(f"  Cost of Debt (after-tax)   : {wacc_breakdown.get('cost_of_debt', 0):>8.2f}%")
        lines.append(f"  Market Cap                 : ${wacc_breakdown.get('market_cap', 0)/1e9:>8.2f}B")
        lines.append(f"  Total Debt                 : ${wacc_breakdown.get('total_debt', 0)/1e9:>8.2f}B")
        lines.append(f"  Equity Weight              : {wacc_breakdown.get('equity_weight', 0):>8.1f}%")
        lines.append(f"  Debt Weight                : {wacc_breakdown.get('debt_weight', 0):>8.1f}%")
        lines.append(f"  ═══════════════════════════════════════════════════════════")
        lines.append(f"  WACC                       : {wacc_breakdown.get('wacc', 0):>8.2f}%")

        # Section 3: FCF Projections
        lines.append(f"\n{'─'*100}")
        lines.append("  📈 FREE CASH FLOW PROJECTIONS & DISCOUNTING")
        lines.append(f"{'─'*100}")
        lines.append(f"  {'Year':<8} {'Projected FCF':>15} {'Discount Factor':>18} {'Present Value':>18}")
        lines.append(f"  {'-'*8} {'-'*15} {'-'*18} {'-'*18}")

        for proj in projections:
            year = proj.get("year", 0)
            fcf = proj.get("fcf", 0)
            discount_factor = proj.get("discount_factor", 0)
            pv = proj.get("pv_fcf", 0)
            lines.append(f"  Year {year:<3} ${fcf/1e9:>12.2f}B   " f"{discount_factor:>16.4f}   ${pv/1e9:>15.2f}B")

        lines.append(f"  {'-'*8} {'-'*15} {'-'*18} {'-'*18}")
        total_pv_fcf = sum(p.get("pv_fcf", 0) for p in projections)
        lines.append(f"  {'TOTAL':<8} {' '*15} {' '*18} ${total_pv_fcf/1e9:>15.2f}B")

        # Section 4: Terminal Value
        lines.append(f"\n{'─'*100}")
        lines.append("  🎯 TERMINAL VALUE")
        lines.append(f"{'─'*100}")
        final_year_fcf = projections[-1].get("fcf", 0) if projections else 0
        lines.append(f"  Final Year FCF (Year {len(projections)})    : ${final_year_fcf/1e9:>8.2f}B")
        lines.append(f"  Terminal Growth Rate       : {terminal.get('terminal_growth', 0):>8.2f}%")
        lines.append(
            f"  Terminal FCF (perpetuity)  : ${final_year_fcf * (1 + terminal.get('terminal_growth', 0)/100) / 1e9:>8.2f}B"
        )
        lines.append(
            f"  Terminal Value             : ${terminal.get('terminal_value', 0)/1e9:>8.2f}B  (FCF / (WACC - g))"
        )
        lines.append(f"  Discount Factor (Year {len(projections)})   : {terminal.get('discount_factor', 0):>8.4f}")
        lines.append(f"  Present Value (Terminal)   : ${terminal.get('pv_terminal', 0)/1e9:>8.2f}B")

        # Section 5: Valuation Summary
        lines.append(f"\n{'─'*100}")
        lines.append("  💎 VALUATION SUMMARY")
        lines.append(f"{'─'*100}")
        lines.append(f"  PV of Projected FCF        : ${valuation.get('pv_fcf', 0)/1e9:>8.2f}B")
        lines.append(f"  PV of Terminal Value       : ${valuation.get('pv_terminal', 0)/1e9:>8.2f}B")
        lines.append(f"  ═══════════════════════════════════════════════════════════")
        lines.append(f"  Enterprise Value           : ${valuation.get('enterprise_value', 0)/1e9:>8.2f}B")
        lines.append(f"  Less: Net Debt             : ${valuation.get('net_debt', 0)/1e9:>8.2f}B")
        lines.append(f"  Equity Value               : ${valuation.get('equity_value', 0)/1e9:>8.2f}B")
        lines.append(f"  Shares Outstanding         : {valuation.get('shares_outstanding', 0)/1e9:>8.2f}B")
        lines.append(f"  ═══════════════════════════════════════════════════════════")
        lines.append(f"  DCF Fair Value per Share   : ${valuation.get('fair_value', 0):>8.2f}")
        lines.append(f"  Current Price              : ${valuation.get('current_price', 0):>8.2f}")
        lines.append(f"  Upside / (Downside)        : {valuation.get('upside_pct', 0):>+8.1f}%")

        lines.append(f"{'='*100}\n")

        return "\n".join(lines)

    @staticmethod
    def format_relative_valuation_table(symbol: str, models: List[Dict[str, Any]], current_price: float) -> str:
        """
        Format relative valuation models (P/E, P/S, P/B, EV/EBITDA) as single table.

        Args:
            symbol: Stock symbol
            models: List of model dicts with keys: name, metric_name, metric_value,
                   sector_multiple, sector_median_label, fair_value, confidence, applicable, reason
            current_price: Current stock price

        Returns:
            Formatted ASCII table string
        """
        lines = []
        lines.append(f"\n{'='*120}")
        lines.append(f"  RELATIVE VALUATION - {symbol}  (Current Price: ${current_price:.2f})")
        lines.append(f"{'='*120}")

        lines.append(
            f"\n  {'Model':<12} {'Metric':<20} {'Value':>12} {'Sector Multiple':>18} {'Fair Value':>15} {'vs Current':>12} {'Conf':>6}"
        )
        lines.append(f"  {'-'*12} {'-'*20} {'-'*12} {'-'*18} {'-'*15} {'-'*12} {'-'*6}")

        for model in models:
            name = model.get("name", "N/A")
            metric_name = model.get("metric_name", "N/A")
            metric_value = model.get("metric_value", 0)
            sector_multiple = model.get("sector_multiple", 0)
            sector_label = model.get("sector_median_label", "N/A")
            fair_value = model.get("fair_value", 0)
            confidence = model.get("confidence", 0)
            applicable = model.get("applicable", True)
            reason = model.get("reason", "")

            if not applicable:
                lines.append(f"  {name:<12} {'NOT APPLICABLE':<20} {' '*12} {' '*18} {' '*15} {' '*12} {' '*6}")
                lines.append(f"  {' '*12} └─ Reason: {reason}")
            else:
                # Format metric value (billions if > 1B, millions otherwise)
                if metric_value > 1e9:
                    metric_str = f"${metric_value/1e9:.2f}B"
                elif metric_value > 1e6:
                    metric_str = f"${metric_value/1e6:.1f}M"
                else:
                    metric_str = f"{metric_value:.2f}"

                # Calculate upside/downside
                upside_pct = ((fair_value - current_price) / current_price) * 100 if current_price > 0 else 0
                upside_str = f"{upside_pct:+.1f}%"

                lines.append(
                    f"  {name:<12} {metric_name:<20} {metric_str:>12} "
                    f"{sector_multiple:>17.1f}x  ${fair_value:>13.2f}  {upside_str:>11}  {confidence:>5.0f}%"
                )
                lines.append(f"  {' '*12} └─ Sector: {sector_label}")

        lines.append(f"{'='*120}\n")

        return "\n".join(lines)

    @staticmethod
    def format_ggm_table(
        symbol: str,
        inputs: Dict[str, Any],
        dividend_projections: List[Dict[str, float]],
        valuation: Dict[str, float],
        applicable: bool = True,
        reason: str = "",
    ) -> str:
        """
        Format Gordon Growth Model valuation table.

        Args:
            symbol: Stock symbol
            inputs: dividend_per_share, payout_ratio, cost_of_equity, dividend_growth_rate,
                   historical_dividends (list of {quarter, dividend})
            dividend_projections: List of {year, dividend, discount_factor, pv_dividend}
            valuation: pv_dividends, terminal_value, pv_terminal, fair_value, current_price, upside_pct
            applicable: Whether GGM is applicable
            reason: Reason if not applicable

        Returns:
            Formatted ASCII table string
        """
        lines = []
        lines.append(f"\n{'='*100}")
        lines.append(f"  GORDON GROWTH MODEL (GGM) - {symbol}")
        lines.append(f"{'='*100}")

        if not applicable:
            lines.append(f"\n  ❌ GGM NOT APPLICABLE")
            lines.append(f"  Reason: {reason}")
            lines.append(f"{'='*100}\n")
            return "\n".join(lines)

        # Section 1: Historical Dividends
        lines.append(f"\n{'─'*100}")
        lines.append("  📊 HISTORICAL DIVIDENDS (Last 12 Quarters)")
        lines.append(f"{'─'*100}")
        lines.append(f"  {'Quarter':<15} {'Dividend per Share':>20}")
        lines.append(f"  {'-'*15} {'-'*20}")

        historical = inputs.get("historical_dividends", [])
        for div in historical[-12:]:  # Last 12 quarters
            quarter = div.get("quarter", "N/A")
            dividend = div.get("dividend", 0)
            lines.append(f"  {quarter:<15} ${dividend:>19.4f}")

        # Section 2: Inputs
        lines.append(f"\n{'─'*100}")
        lines.append("  💰 GGM INPUTS")
        lines.append(f"{'─'*100}")
        lines.append(f"  Current Dividend (annual)  : ${inputs.get('dividend_per_share', 0):>8.4f}")
        lines.append(f"  Payout Ratio               : {inputs.get('payout_ratio', 0):>8.1f}%")
        lines.append(
            f"  Dividend Growth Rate       : {inputs.get('dividend_growth_rate', 0):>8.2f}%  (historical average)"
        )
        lines.append(f"  Cost of Equity             : {inputs.get('cost_of_equity', 0):>8.2f}%")

        # Section 3: Dividend Projections (if using multi-stage)
        if dividend_projections:
            lines.append(f"\n{'─'*100}")
            lines.append("  📈 DIVIDEND PROJECTIONS (Multi-Stage Model)")
            lines.append(f"{'─'*100}")
            lines.append(f"  {'Year':<8} {'Projected Dividend':>20} {'Discount Factor':>18} {'Present Value':>18}")
            lines.append(f"  {'-'*8} {'-'*20} {'-'*18} {'-'*18}")

            for proj in dividend_projections:
                year = proj.get("year", 0)
                dividend = proj.get("dividend", 0)
                discount_factor = proj.get("discount_factor", 0)
                pv = proj.get("pv_dividend", 0)
                lines.append(f"  Year {year:<3} ${dividend:>18.4f}   " f"{discount_factor:>16.4f}   ${pv:>16.2f}")

            lines.append(f"  {'-'*8} {'-'*20} {'-'*18} {'-'*18}")
            total_pv_div = sum(p.get("pv_dividend", 0) for p in dividend_projections)
            lines.append(f"  {'TOTAL':<8} {' '*20} {' '*18} ${total_pv_div:>16.2f}")

        # Section 4: Valuation
        lines.append(f"\n{'─'*100}")
        lines.append("  💎 GGM VALUATION")
        lines.append(f"{'─'*100}")

        if dividend_projections:
            lines.append(f"  PV of Projected Dividends  : ${valuation.get('pv_dividends', 0):>8.2f}")
            lines.append(f"  Terminal Value             : ${valuation.get('terminal_value', 0):>8.2f}")
            lines.append(f"  PV of Terminal Value       : ${valuation.get('pv_terminal', 0):>8.2f}")
            lines.append(f"  ═══════════════════════════════════════════════════════════")

        d1 = inputs.get("dividend_per_share", 0) * (1 + inputs.get("dividend_growth_rate", 0) / 100)
        r_minus_g = inputs.get("cost_of_equity", 0) - inputs.get("dividend_growth_rate", 0)
        lines.append(f"  Formula: V = D₁ / (r - g)")
        lines.append(f"           D₁ = ${d1:.4f}  (next year dividend)")
        lines.append(f"           r  = {inputs.get('cost_of_equity', 0):.2f}%  (cost of equity)")
        lines.append(f"           g  = {inputs.get('dividend_growth_rate', 0):.2f}%  (growth rate)")
        lines.append(f"           r - g = {r_minus_g:.2f}%")
        lines.append(f"  ═══════════════════════════════════════════════════════════")
        lines.append(f"  GGM Fair Value per Share   : ${valuation.get('fair_value', 0):>8.2f}")
        lines.append(f"  Current Price              : ${valuation.get('current_price', 0):>8.2f}")
        lines.append(f"  Upside / (Downside)        : {valuation.get('upside_pct', 0):>+8.1f}%")

        lines.append(f"{'='*100}\n")

        return "\n".join(lines)

    @staticmethod
    def format_valuation_summary_table(
        symbol: str,
        all_models: List[Dict[str, Any]],
        dynamic_weights: Dict[str, float],
        blended_fair_value: float,
        current_price: float,
        tier: str,
        notes: Optional[List[str]] = None,
    ) -> str:
        """
        Format comprehensive valuation summary with all models.

        Args:
            symbol: Stock symbol
            all_models: List of {name, fair_value, confidence, weight, applicable}
            dynamic_weights: Dict of model_name -> weight (%)
            blended_fair_value: Weighted average fair value
            current_price: Current stock price
            tier: Dynamic weighting tier classification
            notes: Optional diagnostic notes from weighting/orchestrator

        Returns:
            Formatted ASCII table string
        """
        notes = [note for note in (notes or []) if note]

        lines = []
        lines.append(f"\n{'='*100}")
        lines.append(f"  VALUATION SUMMARY - {symbol}")
        lines.append(f"{'='*100}")

        lines.append(f"\n  {'Model':<15} {'Fair Value':>15} {'Confidence':>12} {'Weight':>10} {'Weighted FV':>15}")
        lines.append(f"  {'-'*15} {'-'*15} {'-'*12} {'-'*10} {'-'*15}")

        total_weighted_fv = 0.0
        total_weight = 0.0

        for model in all_models:
            name = model.get("name", "N/A")
            fair_value = model.get("fair_value", 0)
            confidence = model.get("confidence", 0)
            weight = model.get("weight", 0)
            applicable = model.get("applicable", True)

            if not applicable or weight == 0:
                lines.append(f"  {name:<15} {'-':>15} {'-':>12} {weight:>9.0f}%  {'-':>15}")
            else:
                weighted_fv = fair_value * (weight / 100)
                total_weighted_fv += weighted_fv
                total_weight += weight

                lines.append(
                    f"  {name:<15} ${fair_value:>14.2f}  {confidence:>10.0f}%  "
                    f"{weight:>9.0f}%  ${weighted_fv:>14.2f}"
                )

        lines.append(f"  {'-'*15} {'-'*15} {'-'*12} {'-'*10} {'-'*15}")
        lines.append(f"  {'BLENDED':<15} {' '*15} {' '*12} {total_weight:>9.0f}%  " f"${total_weighted_fv:>14.2f}")

        upside_pct = ((blended_fair_value - current_price) / current_price) * 100 if current_price > 0 else 0

        lines.append(f"\n{'─'*100}")
        lines.append(f"  Tier Classification        : {tier}")
        lines.append(f"  Blended Fair Value         : ${blended_fair_value:>8.2f}")
        lines.append(f"  Current Price              : ${current_price:>8.2f}")
        lines.append(f"  Upside / (Downside)        : {upside_pct:>+8.1f}%")
        if notes:
            lines.append(f"\n{'─'*100}")
            lines.append("  Notes / Diagnostics")
            lines.append(f"{'─'*100}")
            for note in notes:
                lines.append(f"  • {note}")

        lines.append(f"{'='*100}\n")

        return "\n".join(lines)
