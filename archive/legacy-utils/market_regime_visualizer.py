#!/usr/bin/env python3
"""
Market Regime Visualizer
Reusable component for generating market regime visualizations
Used across all stock analyses for consistent reporting
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try to import rich for terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.warning("Rich library not available - terminal output will be basic")

# Try to import reportlab for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import Table as RLTable, TableStyle, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available - PDF components will be skipped")


class MarketRegimeVisualizer:
    """
    Generate visual representations of market regime for multiple output formats:
    - Terminal (ASCII art and rich tables)
    - HTML (for web reports)
    - PDF (for report generation)
    - JSON (for API responses)
    """

    def __init__(self):
        """Initialize visualizer"""
        self.console = Console() if RICH_AVAILABLE else None
        self.styles = getSampleStyleSheet() if REPORTLAB_AVAILABLE else None

    def generate_ascii_art(self, regime: str) -> str:
        """
        Generate ASCII art representation of market regime

        Args:
            regime: Market regime (risk_on, risk_off, mixed)

        Returns:
            ASCII art string
        """
        ascii_art = {
            "risk_on": """
    📈 RISK-ON MARKET REGIME 📈
    ╔══════════════════════════╗
    ║    🚀 BULLISH MODE 🚀    ║
    ║  ┌─────────────────────┐ ║
    ║  │ ▲ Stocks > Bonds    │ ║
    ║  │ ▲ Small > Large Cap │ ║
    ║  │ ▲ Commodities Up    │ ║
    ║  │ ▼ Gold Declining    │ ║
    ║  │ ▼ VIX < 20         │ ║
    ║  └─────────────────────┘ ║
    ╚══════════════════════════╝""",
            "risk_off": """
    📉 RISK-OFF MARKET REGIME 📉
    ╔══════════════════════════╗
    ║    🛡️ DEFENSIVE MODE 🛡️   ║
    ║  ┌─────────────────────┐ ║
    ║  │ ▲ Bonds > Stocks    │ ║
    ║  │ ▲ Gold Rally        │ ║
    ║  │ ▲ VIX > 30         │ ║
    ║  │ ▼ Oil Declining     │ ║
    ║  │ ▼ Small Caps Weak   │ ║
    ║  └─────────────────────┘ ║
    ╚══════════════════════════╝""",
            "mixed": """
    ⚖️ MIXED MARKET REGIME ⚖️
    ╔══════════════════════════╗
    ║    🔄 NEUTRAL MODE 🔄    ║
    ║  ┌─────────────────────┐ ║
    ║  │ ~ Mixed Signals     │ ║
    ║  │ ~ No Clear Trend    │ ║
    ║  │ ~ Sector Rotation   │ ║
    ║  │ ~ Await Catalyst    │ ║
    ║  └─────────────────────┘ ║
    ╚══════════════════════════╝""",
        }

        return ascii_art.get(regime, ascii_art["mixed"])

    def create_terminal_table(self, market_data: Dict) -> Optional[Any]:
        """
        Create rich terminal table for market performance

        Args:
            market_data: Market context data

        Returns:
            Rich Table object or None
        """
        if not RICH_AVAILABLE:
            return self._create_basic_table(market_data)

        from rich.table import Table

        table = Table(title="📊 Market Performance Analysis", show_header=True, header_style="bold magenta")

        table.add_column("Asset Class", style="cyan", width=20)
        table.add_column("ETF", style="yellow", width=8)
        table.add_column("1M Return", justify="right", style="green")
        table.add_column("Volatility", justify="right", style="blue")
        table.add_column("Signal", justify="center", width=15)

        # Add rows based on market data
        medium_term = market_data.get("medium_term", {})

        # Process each asset class
        asset_classes = [
            ("S&P 500", "broad_market", "SPY"),
            ("Aggregate Bonds", "bonds", "AGG"),
            ("Small Caps", "small_cap", "IWM"),
            ("Gold", "gold", "GLD"),
            ("Silver", "silver", "SLV"),
            ("Crude Oil", "oil", "USO"),
            ("Commodities", "commodities", "DBC"),
            ("International", "international", "EFA"),
            ("Emerging Markets", "emerging", "EEM"),
        ]

        for name, key, etf in asset_classes:
            data = medium_term.get(key, {})
            if data:
                signal = self._get_signal(key, data.get("return", 0))
                table.add_row(name, etf, f"{data.get('return', 0):.2%}", f"{data.get('volatility', 0):.2%}", signal)

        return table

    def _create_basic_table(self, market_data: Dict) -> str:
        """Create basic text table for non-rich environments"""
        lines = []
        lines.append("=" * 60)
        lines.append("MARKET PERFORMANCE ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"{'Asset Class':<20} {'ETF':<8} {'1M Return':>10} {'Signal':<15}")
        lines.append("-" * 60)

        medium_term = market_data.get("medium_term", {})

        asset_classes = [
            ("S&P 500", "broad_market", "SPY"),
            ("Bonds", "bonds", "AGG"),
            ("Gold", "gold", "GLD"),
            ("Oil", "oil", "USO"),
        ]

        for name, key, etf in asset_classes:
            data = medium_term.get(key, {})
            if data:
                signal = self._get_signal(key, data.get("return", 0))
                lines.append(f"{name:<20} {etf:<8} {data.get('return', 0):>10.2%} {signal:<15}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _get_signal(self, asset_type: str, return_value: float) -> str:
        """Determine signal based on asset type and return"""
        signals = {
            "broad_market": (
                "🟢 Risk-On" if return_value > 0.03 else "🔴 Risk-Off" if return_value < -0.03 else "⚪ Neutral"
            ),
            "bonds": (
                "🔴 Flight-to-Safety" if return_value > 0.02 else "🟢 Risk-On" if return_value < -0.02 else "⚪ Neutral"
            ),
            "gold": (
                "🔴 Safe Haven" if return_value > 0.05 else "🟢 Risk-On" if return_value < -0.02 else "⚪ Neutral"
            ),
            "oil": ("🟢 Growth" if return_value > 0.03 else "🔴 Slowdown" if return_value < -0.05 else "⚪ Stable"),
            "commodities": (
                "🟢 Inflation" if return_value > 0.03 else "🔴 Deflation" if return_value < -0.05 else "⚪ Neutral"
            ),
        }

        return signals.get(asset_type, "⚪ Neutral")

    def create_pdf_components(self, market_data: Dict, sector_data: Optional[Dict] = None) -> List[Any]:
        """
        Create PDF-ready components for market regime visualization

        Args:
            market_data: Market context data
            sector_data: Sector performance data

        Returns:
            List of ReportLab flowable objects
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available - cannot create PDF components")
            return []

        components = []

        # Add market regime header
        regime = market_data.get("market_regime", "neutral")
        regime_style = self._get_regime_style(regime)

        # Create regime header paragraph
        regime_text = self._get_regime_description(regime)
        p = Paragraph(regime_text, regime_style)
        components.append(p)

        # Create market performance table
        market_table = self._create_pdf_market_table(market_data)
        if market_table:
            components.append(market_table)

        # Create sector performance table if provided
        if sector_data:
            sector_table = self._create_pdf_sector_table(sector_data)
            if sector_table:
                components.append(sector_table)

        # Add risk signals summary
        risk_signals = self._create_risk_signals_summary(market_data)
        if risk_signals:
            components.append(risk_signals)

        return components

    def _get_regime_style(self, regime: str) -> Any:
        """Get ReportLab style for regime header"""
        if not REPORTLAB_AVAILABLE:
            return None

        style = ParagraphStyle(
            "RegimeHeader",
            parent=self.styles["Heading1"],
            fontSize=16,
            textColor=colors.green if regime == "risk_on" else colors.red if regime == "risk_off" else colors.blue,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
        return style

    def _get_regime_description(self, regime: str) -> str:
        """Get descriptive text for market regime"""
        descriptions = {
            "risk_on": "<b>🚀 RISK-ON MARKET REGIME</b><br/>Markets favor growth assets with strong risk appetite",
            "risk_off": "<b>🛡️ RISK-OFF MARKET REGIME</b><br/>Markets favor defensive assets with flight to safety",
            "mixed": "<b>🔄 MIXED MARKET REGIME</b><br/>Markets show mixed signals with no clear directional bias",
        }
        return descriptions.get(regime, descriptions["mixed"])

    def _create_pdf_market_table(self, market_data: Dict) -> Optional[Any]:
        """Create PDF table for market performance"""
        if not REPORTLAB_AVAILABLE:
            return None

        medium_term = market_data.get("medium_term", {})
        if not medium_term:
            return None

        # Table data
        data = [["Asset Class", "ETF", "1M Return", "Volatility", "Signal"]]

        asset_classes = [
            ("S&P 500", "broad_market", "SPY"),
            ("Bonds", "bonds", "AGG"),
            ("Small Caps", "small_cap", "IWM"),
            ("Gold", "gold", "GLD"),
            ("Oil", "oil", "USO"),
            ("Commodities", "commodities", "DBC"),
        ]

        for name, key, etf in asset_classes:
            asset_data = medium_term.get(key, {})
            if asset_data:
                signal = self._get_signal_text(key, asset_data.get("return", 0))
                data.append(
                    [name, etf, f"{asset_data.get('return', 0):.2%}", f"{asset_data.get('volatility', 0):.2%}", signal]
                )

        # Create table
        table = RLTable(data, colWidths=[2 * inch, 0.8 * inch, 1 * inch, 1 * inch, 1.5 * inch])

        # Apply style
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        return table

    def _create_pdf_sector_table(self, sector_data: Dict) -> Optional[Any]:
        """Create PDF table for sector performance"""
        if not REPORTLAB_AVAILABLE or not sector_data:
            return None

        # Table data
        data = [["Sector", "ETF", "1M Return", "Rel. Strength", "Rank"]]

        sectors = sector_data.get("sectors", {})
        rankings = sector_data.get("rankings", [])

        for rank_data in rankings[:10]:  # Top 10 sectors
            sector = rank_data["sector"]
            sector_info = sectors.get(sector, {})

            data.append(
                [
                    sector.replace("_", " ").title(),
                    sector_info.get("etf", ""),
                    f"{rank_data.get('return', 0):.2%}",
                    f"{rank_data.get('relative_strength', 0):.2%}",
                    str(rank_data.get("rank", "")),
                ]
            )

        # Create table
        table = RLTable(data, colWidths=[2 * inch, 0.8 * inch, 1 * inch, 1.2 * inch, 0.6 * inch])

        # Apply style
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.lightblue),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        return table

    def _create_risk_signals_summary(self, market_data: Dict) -> Optional[Any]:
        """Create risk signals summary paragraph"""
        if not REPORTLAB_AVAILABLE:
            return None

        risk_signals = market_data.get("risk_signals", {})
        if not risk_signals:
            return None

        # Build summary text
        signal_texts = []
        for signal_type, signal_value in risk_signals.items():
            signal_texts.append(f"<b>{signal_type.replace('_', ' ').title()}:</b> {signal_value}")

        summary_text = "<b>Risk Signals Summary</b><br/>" + "<br/>".join(signal_texts)

        style = ParagraphStyle("RiskSummary", parent=self.styles["Normal"], fontSize=10, leftIndent=20, spaceAfter=12)

        return Paragraph(summary_text, style)

    def _get_signal_text(self, asset_type: str, return_value: float) -> str:
        """Get text signal without emojis for PDF"""
        signals = {
            "broad_market": ("Risk-On" if return_value > 0.03 else "Risk-Off" if return_value < -0.03 else "Neutral"),
            "bonds": ("Flight-to-Safety" if return_value > 0.02 else "Risk-On" if return_value < -0.02 else "Neutral"),
            "gold": ("Safe Haven" if return_value > 0.05 else "Risk-On" if return_value < -0.02 else "Neutral"),
            "oil": ("Growth" if return_value > 0.03 else "Slowdown" if return_value < -0.05 else "Stable"),
            "commodities": ("Inflation" if return_value > 0.03 else "Deflation" if return_value < -0.05 else "Neutral"),
        }

        return signals.get(asset_type, "Neutral")

    def generate_html_components(self, market_data: Dict, sector_data: Optional[Dict] = None) -> str:
        """
        Generate HTML components for web reports

        Args:
            market_data: Market context data
            sector_data: Sector performance data

        Returns:
            HTML string
        """
        html_parts = []

        # Market regime header
        regime = market_data.get("market_regime", "neutral")
        regime_color = "#28a745" if regime == "risk_on" else "#dc3545" if regime == "risk_off" else "#007bff"

        html_parts.append(
            f"""
        <div style="text-align: center; padding: 20px; background-color: {regime_color}; color: white; border-radius: 10px;">
            <h2>{self._get_regime_title(regime)}</h2>
            <p>{self._get_regime_subtitle(regime)}</p>
        </div>
        """
        )

        # Market performance table
        html_parts.append(self._create_html_market_table(market_data))

        # Sector performance if provided
        if sector_data:
            html_parts.append(self._create_html_sector_table(sector_data))

        # Risk signals
        risk_signals = market_data.get("risk_signals", {})
        if risk_signals:
            html_parts.append(self._create_html_risk_signals(risk_signals))

        return "\n".join(html_parts)

    def _get_regime_title(self, regime: str) -> str:
        """Get regime title for HTML"""
        titles = {
            "risk_on": "🚀 RISK-ON MARKET REGIME",
            "risk_off": "🛡️ RISK-OFF MARKET REGIME",
            "mixed": "🔄 MIXED MARKET REGIME",
        }
        return titles.get(regime, titles["mixed"])

    def _get_regime_subtitle(self, regime: str) -> str:
        """Get regime subtitle for HTML"""
        subtitles = {
            "risk_on": "Markets favor growth assets with strong risk appetite",
            "risk_off": "Markets favor defensive assets with flight to safety",
            "mixed": "Markets show mixed signals with no clear directional bias",
        }
        return subtitles.get(regime, subtitles["mixed"])

    def _create_html_market_table(self, market_data: Dict) -> str:
        """Create HTML table for market performance"""
        medium_term = market_data.get("medium_term", {})
        if not medium_term:
            return ""

        html = """
        <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
            <thead>
                <tr style="background-color: #f8f9fa;">
                    <th style="padding: 10px; text-align: left;">Asset Class</th>
                    <th style="padding: 10px; text-align: center;">ETF</th>
                    <th style="padding: 10px; text-align: right;">1M Return</th>
                    <th style="padding: 10px; text-align: right;">Volatility</th>
                    <th style="padding: 10px; text-align: center;">Signal</th>
                </tr>
            </thead>
            <tbody>
        """

        asset_classes = [
            ("S&P 500", "broad_market", "SPY"),
            ("Bonds", "bonds", "AGG"),
            ("Small Caps", "small_cap", "IWM"),
            ("Gold", "gold", "GLD"),
            ("Oil", "oil", "USO"),
            ("Commodities", "commodities", "DBC"),
        ]

        for name, key, etf in asset_classes:
            data = medium_term.get(key, {})
            if data:
                return_val = data.get("return", 0)
                return_color = "#28a745" if return_val > 0 else "#dc3545"
                signal = self._get_signal_text(key, return_val)

                html += f"""
                <tr>
                    <td style="padding: 8px;">{name}</td>
                    <td style="padding: 8px; text-align: center;">{etf}</td>
                    <td style="padding: 8px; text-align: right; color: {return_color};">{return_val:.2%}</td>
                    <td style="padding: 8px; text-align: right;">{data.get('volatility', 0):.2%}</td>
                    <td style="padding: 8px; text-align: center;">{signal}</td>
                </tr>
                """

        html += """
            </tbody>
        </table>
        """

        return html

    def _create_html_sector_table(self, sector_data: Dict) -> str:
        """Create HTML table for sector performance"""
        rankings = sector_data.get("rankings", [])
        if not rankings:
            return ""

        html = """
        <h3>Sector Performance Rankings</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
            <thead>
                <tr style="background-color: #e9ecef;">
                    <th style="padding: 10px; text-align: left;">Rank</th>
                    <th style="padding: 10px; text-align: left;">Sector</th>
                    <th style="padding: 10px; text-align: right;">1M Return</th>
                    <th style="padding: 10px; text-align: right;">Rel. Strength</th>
                </tr>
            </thead>
            <tbody>
        """

        for rank_data in rankings[:10]:
            return_val = rank_data.get("return", 0)
            return_color = "#28a745" if return_val > 0 else "#dc3545"

            html += f"""
            <tr>
                <td style="padding: 8px;">{rank_data.get('rank', '')}</td>
                <td style="padding: 8px;">{rank_data['sector'].replace('_', ' ').title()}</td>
                <td style="padding: 8px; text-align: right; color: {return_color};">{return_val:.2%}</td>
                <td style="padding: 8px; text-align: right;">{rank_data.get('relative_strength', 0):.2%}</td>
            </tr>
            """

        html += """
            </tbody>
        </table>
        """

        return html

    def _create_html_risk_signals(self, risk_signals: Dict) -> str:
        """Create HTML for risk signals"""
        html = """
        <div style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
            <h3>Risk Signals</h3>
            <ul style="list-style-type: none; padding: 0;">
        """

        for signal_type, signal_value in risk_signals.items():
            icon = (
                "✅"
                if "risk_on" in str(signal_value).lower()
                else "⚠️" if "risk_off" in str(signal_value).lower() else "➖"
            )
            html += f"""
            <li style="padding: 5px 0;">
                {icon} <strong>{signal_type.replace('_', ' ').title()}:</strong> {signal_value}
            </li>
            """

        html += """
            </ul>
        </div>
        """

        return html

    def generate_json_summary(self, market_data: Dict, sector_data: Optional[Dict] = None) -> Dict:
        """
        Generate JSON summary for API responses

        Args:
            market_data: Market context data
            sector_data: Sector performance data

        Returns:
            JSON-serializable dictionary
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "market_regime": market_data.get("market_regime", "neutral"),
            "risk_signals": market_data.get("risk_signals", {}),
            "key_metrics": {},
        }

        # Extract key metrics
        medium_term = market_data.get("medium_term", {})

        if "broad_market" in medium_term:
            summary["key_metrics"]["spy_return"] = medium_term["broad_market"].get("return", 0)
            summary["key_metrics"]["spy_volatility"] = medium_term["broad_market"].get("volatility", 0)

        if "bonds" in medium_term:
            summary["key_metrics"]["bond_return"] = medium_term["bonds"].get("return", 0)

        if "gold" in medium_term:
            summary["key_metrics"]["gold_return"] = medium_term["gold"].get("return", 0)

        if "commodities" in medium_term:
            summary["key_metrics"]["commodity_return"] = medium_term["commodities"].get("return", 0)

        # Add sector rankings if available
        if sector_data:
            rankings = sector_data.get("rankings", [])
            summary["top_sectors"] = [
                {"sector": r["sector"], "return": r["return"], "rank": r["rank"]} for r in rankings[:5]
            ]

        # Add interpretation
        summary["interpretation"] = self._generate_interpretation(summary["market_regime"], summary["risk_signals"])

        return summary

    def _generate_interpretation(self, regime: str, signals: Dict) -> str:
        """Generate human-readable interpretation"""
        interpretations = {
            "risk_on": "Market conditions favor growth-oriented investments with higher risk tolerance.",
            "risk_off": "Market conditions favor defensive positioning and capital preservation.",
            "mixed": "Market shows mixed signals; consider balanced positioning with selective opportunities.",
        }

        base_interpretation = interpretations.get(regime, interpretations["mixed"])

        # Add specific signal insights
        if signals.get("commodity") == "inflationary":
            base_interpretation += " Commodity signals suggest inflationary pressures."
        elif signals.get("commodity") == "deflationary":
            base_interpretation += " Commodity weakness suggests deflationary concerns."

        if signals.get("safe_haven") == "risk_off":
            base_interpretation += " Strong safe haven demand indicates elevated market stress."

        return base_interpretation
